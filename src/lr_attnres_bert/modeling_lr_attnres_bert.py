"""LR-AttnRes-BERT: a plain GQA + RoPE + ReLU^2 encoder with Block Sliced Low-Rank
Attention Residuals.

Encoder layer (pre-norm)::

    a = Attention(RMSNorm(read_attn))        # GQA, rotary positions, no bias
    m = MLP(RMSNorm(read_mlp))               # W_o relu(W_i x)^2, no bias

In a standard encoder both reads are the residual sum ``x = emb + sum of all
writes``. LR-AttnRes (Attention Residuals with sliced low-rank routing keys, Su
2026, arXiv:2607.09694) replaces a read with a softmax mix over depth::

    alpha_i = softmax_i( q_t . RMSNorm(tail_r(s_i)) )      q_t in R^r, learned
    h_t     = sum_i alpha_i s_i

where the sources ``s_i`` are the embedding, the sums of the finished groups of
sublayer writes (Block AttnRes: the ``2 L`` writes form ``attnres_blocks`` groups)
and the open group's partial sum, ``tail_r`` takes the last ``r`` channels, and
``q_t`` is a per-read pseudo-query (zero-initialised: a fresh model reads the plain
residual sum, up to the RMSNorm that follows). With the default read sites (the
paper's) every sublayer input and the encoder output are routed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

try:  # transformers >= 5: init helpers that skip tensors a checkpoint filled
    from transformers import initialization as init
except ImportError:  # pragma: no cover
    from torch.nn import init

from .configuration_lr_attnres_bert import LRAttnResBertConfig

_KEY_EPS = 1e-6


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------

class LRAttnResBertRotaryEmbedding(nn.Module):
    """cos / sin tables for rotate-half RoPE, computed on demand.

    Deliberately holds no buffer: transformers >= 5 builds a model on the meta
    device and leaves computed non-persistent buffers zero after loading. The
    tables are memoised per (length, device, dtype) in eager mode and recomputed
    inside a compiled graph, where inductor folds them.
    """

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._cache: dict = {}

    def forward(self, seq_len: int, device, dtype):
        compiling = torch.compiler.is_compiling()
        key = (seq_len, str(device), dtype)
        if not compiling and key in self._cache:
            return self._cache[key]
        # Built outside inference mode so a cached table can later be saved for
        # backward (fine-tuning after an inference_mode call).
        with torch.inference_mode(False), torch.no_grad():
            inv_freq = 1.0 / (self.theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim))
            t = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
        if not compiling:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = (cos, sin)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` (B, H, T, D); ``cos`` / ``sin`` (T, D)."""
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class LRAttnResBertRMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` whose gain follows the input's dtype.

    Under autocast the weights stay fp32 while e.g. the prediction head's input is
    bf16; ``rms_norm`` is not on autocast's fp32 list, so a plain ``nn.RMSNorm``
    then warns and falls back to an unfused implementation. Casting the gain keeps
    the fused kernel (which accumulates in fp32 either way); in pure bf16 / fp32
    inference the cast is a no-op.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class LRAttnResBertAttention(nn.Module):
    """Bidirectional grouped-query attention with rotary positions (SDPA)."""

    def __init__(self, config: LRAttnResBertConfig):
        super().__init__()
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(config.hidden_size,
                              (self.n_head + 2 * self.n_kv_head) * self.head_dim, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_mask=None):
        B, T, C = x.shape
        qkv = self.Wqkv(x).view(B, T, self.n_head + 2 * self.n_kv_head, self.head_dim)
        q, k, v = qkv.split([self.n_head, self.n_kv_head, self.n_kv_head], dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if self.n_kv_head != self.n_head:
            g = self.n_head // self.n_kv_head
            k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.Wo(y.transpose(1, 2).reshape(B, T, C))


class LRAttnResBertMLP(nn.Module):
    """``W_o relu(W_i x)^2`` (squared ReLU, no gate, no bias)."""

    def __init__(self, config: LRAttnResBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.Wo(F.relu(self.Wi(x)).square())


class LRAttnResBertQuery(nn.Module):
    """The pseudo-query of one LR-AttnRes read (zero-initialised)."""

    def __init__(self, rank: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(rank))

    def extra_repr(self) -> str:
        return f"rank={self.query.numel()}"


def _group_ends(config: LRAttnResBertConfig, layer_idx: int) -> tuple:
    """Whether this layer's (attention, MLP) write closes a Block-AttnRes group: the
    ``M = 2 L`` writes form ``N`` groups, group ``j`` ending at write
    ``ceil(M j / N)`` (1-based)."""
    M = 2 * config.num_hidden_layers
    N = config.attnres_blocks
    if N <= 0 or N >= M:
        return True, True
    ends = {-(-M * j // N) for j in range(1, N + 1)}
    return 2 * layer_idx + 1 in ends, 2 * layer_idx + 2 in ends


class LRAttnResBertLayer(nn.Module):
    def __init__(self, config: LRAttnResBertConfig, layer_idx: int):
        super().__init__()
        eps = config.norm_eps
        self.attn_norm = LRAttnResBertRMSNorm(config.hidden_size, eps=eps)
        self.attn = LRAttnResBertAttention(config)
        self.mlp_norm = LRAttnResBertRMSNorm(config.hidden_size, eps=eps)
        self.mlp = LRAttnResBertMLP(config)
        rank = min(config.attnres_rank, config.hidden_size)
        # The first attention read sees the embedding alone: nothing to route.
        self.res_attn = LRAttnResBertQuery(rank) if rank > 0 and layer_idx > 0 else None
        self.res_mlp = (LRAttnResBertQuery(rank)
                        if rank > 0 and config.attnres_read_sites == "all" else None)
        self.closes = _group_ends(config, layer_idx) if rank > 0 else (False, False)


# ---------------------------------------------------------------------------
# LR-AttnRes stream
# ---------------------------------------------------------------------------

def _acc(dtype: torch.dtype) -> torch.dtype:
    """Routing precision: fp32, or the model's dtype when that is wider."""
    return torch.promote_types(dtype, torch.float32)


def _routing_key(s: torch.Tensor, rank: int) -> torch.Tensor:
    """RMSNorm (no gain) of the last ``rank`` channels, at routing precision."""
    return F.rms_norm(s[..., -rank:].to(_acc(s.dtype)), (rank,), eps=_KEY_EPS)


class _Stream:
    """The residual state of one forward pass, evaluated in two phases.

    Between two group closes the closed sources do not change and the
    pseudo-queries are static, so when a group closes ONE batched matmul mixes the
    closed sources for every routed read up to the next close (``hc``, with the
    log-sum-exp of their logits); each read then only merges in the open group's
    partial sum -- an online-softmax step::

        h = hc + sigmoid(q . key(partial) - lse) * (partial - hc)

    which is exactly ``softmax`` over the closed sources and the partial. ``total``
    is the plain residual sum, read by the unrouted (MLP) inputs.

    Reads are numbered in execution order; ``read_pos[t]`` is the number of writes
    before read ``t`` and ``bounds`` the writes after which a group closes (0 = the
    embedding).

    Under autocast (mixed-precision fine-tuning) every source is kept in the
    embedding's dtype and the routing runs with autocast off, so group sums and
    routing logits are never rounded to bf16; in pure bf16 / fp32 inference both are
    no-ops.
    """

    def __init__(self, x, queries, read_pos, bounds, rank):
        self.queries, self.read_pos, self.bounds, self.rank = queries, read_pos, bounds, rank
        self.sources = [x]
        self.keys = [_routing_key(x, rank)]
        self.partial = None
        self.total = x
        self._segment(0)

    def _first_read(self, w):
        return next((t for t, pos in enumerate(self.read_pos) if pos >= w), len(self.read_pos))

    def _segment(self, w):
        """Mix the closed sources for the reads from write ``w`` to the next close."""
        f = self._first_read(w)
        later = [b for b in self.bounds if b > w]
        e = self._first_read(later[0]) if later else len(self.read_pos)
        self.seg_start, self.hc, self.lse = f, None, None
        if e == f:
            return
        V = torch.stack(self.sources, dim=-2)                        # (B, T, S, C)
        K = torch.stack(self.keys, dim=-2)                           # (B, T, S, r) fp32
        with torch.autocast(V.device.type, enabled=False):
            # One GEMM over every (token, source) row, then a materialised transpose:
            # a transposed *view* here trips AOT autograd's stride check in backward
            # under dynamic shapes (varlen fine-tuning batches).
            L = (K @ self.queries[f:e].to(K.dtype).t()).transpose(-1, -2).contiguous()
            self.lse = torch.logsumexp(L, dim=-1)                    # (B, T, R)
            alpha = (L - self.lse[..., None]).exp().to(V.dtype)
            self.hc = alpha @ V                                      # (B, T, R, C)

    def write(self, o, close: bool, w: int):
        """Add write ``w`` (1-based) to the open group and the plain sum; a closing
        write moves the group into the closed sources."""
        o = o.to(self.total.dtype)
        self.total = self.total + o
        self.partial = o if self.partial is None else self.partial + o
        if close:
            self.sources.append(self.partial)
            self.keys.append(_routing_key(self.partial, self.rank))
            self.partial = None
            self._segment(w)

    def read(self, t: int, query) -> torch.Tensor:
        hc = self.hc[..., t - self.seg_start, :]
        if self.partial is None:
            return hc
        p = self.partial
        with torch.autocast(p.device.type, enabled=False):
            k_p = _routing_key(p, self.rank)
            a = torch.sigmoid(k_p @ query.to(k_p.dtype) - self.lse[..., t - self.seg_start])
            return torch.lerp(hc, p, a.unsqueeze(-1).to(p.dtype))   # hc + a (p - hc)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LRAttnResBertPreTrainedModel(PreTrainedModel):
    config_class = LRAttnResBertConfig
    base_model_prefix = "model"
    _no_split_modules = ["LRAttnResBertLayer"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        out_std = std / math.sqrt(2.0 * self.config.num_hidden_layers)
        if isinstance(module, LRAttnResBertAttention):
            init.normal_(module.Wqkv.weight, mean=0.0, std=std)
            init.normal_(module.Wo.weight, mean=0.0, std=out_std)
        elif isinstance(module, LRAttnResBertMLP):
            init.normal_(module.Wi.weight, mean=0.0, std=std)
            init.normal_(module.Wo.weight, mean=0.0, std=out_std)
        elif isinstance(module, LRAttnResBertQuery):
            init.zeros_(module.query)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
            if module.weight is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            # prediction head, decoder bias, classifier (the tied decoder weight is
            # the embedding and is filled through it)
            if module is not getattr(self, "decoder", None):
                init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)


class LRAttnResBertModel(LRAttnResBertPreTrainedModel):
    """The encoder: token embedding -> layers -> final RMSNorm."""

    def __init__(self, config: LRAttnResBertConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LRAttnResBertLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_norm = LRAttnResBertRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary = LRAttnResBertRotaryEmbedding(
            config.hidden_size // config.num_attention_heads, config.rope_theta)
        rank = min(config.attnres_rank, config.hidden_size)
        self.attnres_rank = rank
        self.res_out = LRAttnResBertQuery(rank) if rank > 0 else None
        # Static read schedule: writes before each routed read (attention inputs,
        # MLP inputs too under read_sites="all", then the output) and the writes
        # after which a group closes.
        M = 2 * config.num_hidden_layers
        step = 1 if config.attnres_read_sites == "all" else 2
        self._read_pos = (*range(0, M, step), M)
        self._bounds = (0, *(2 * i + 1 + k for i, layer in enumerate(self.layers)
                             for k in (0, 1) if layer.closes[k]))
        for i, layer in enumerate(self.layers):
            layer.read_idx = (i * 2 // step, (i * 2 + 1) // step)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @staticmethod
    def _attention_mask(attention_mask):
        """(B, T) 0/1 key mask -> SDPA boolean mask, or None when nothing is padded
        (which lets SDPA take its flash kernel)."""
        if attention_mask is None:
            return None
        if not torch.compiler.is_compiling() and bool(attention_mask.all()):
            return None
        return attention_mask.bool()[:, None, None, :]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """``output_hidden_states`` returns the embedding and the plain residual sum
        after every layer (what an unrouted read sees)."""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        x = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        cos, sin = self.rotary(x.shape[1], x.device, x.dtype)
        mask = self._attention_mask(attention_mask)
        states = [x] if output_hidden_states else None

        if self.res_out is None:  # plain residual
            for layer in self.layers:
                x = x + layer.attn(layer.attn_norm(x), cos, sin, mask)
                x = x + layer.mlp(layer.mlp_norm(x))
                if states is not None:
                    states.append(x)
        else:
            rows = [m.query for layer in self.layers
                    for m in (layer.res_attn, layer.res_mlp) if m is not None]
            queries = torch.stack([torch.zeros_like(self.res_out.query), *rows,
                                   self.res_out.query])    # row 0: read 0 has no query
            s = _Stream(x, queries, self._read_pos, self._bounds, self.attnres_rank)
            for i, layer in enumerate(self.layers):
                q = None if layer.res_attn is None else layer.res_attn.query
                h = s.read(layer.read_idx[0], q)
                s.write(layer.attn(layer.attn_norm(h), cos, sin, mask), layer.closes[0],
                        2 * i + 1)
                h = (s.total if layer.res_mlp is None
                     else s.read(layer.read_idx[1], layer.res_mlp.query))
                s.write(layer.mlp(layer.mlp_norm(h)), layer.closes[1], 2 * i + 2)
                if states is not None:
                    states.append(s.total)
            x = s.read(len(self._read_pos) - 1, self.res_out.query)

        x = self.final_norm(x)
        hidden_states = tuple(states) if states is not None else None
        if not return_dict:
            return tuple(v for v in (x, hidden_states) if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=hidden_states)


class LRAttnResBertPredictionHead(nn.Module):
    def __init__(self, config: LRAttnResBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = LRAttnResBertRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x):
        return self.norm(F.gelu(self.dense(x)))


class LRAttnResBertForMaskedLM(LRAttnResBertPreTrainedModel):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.weight"}

    def __init__(self, config: LRAttnResBertConfig):
        super().__init__(config)
        self.model = LRAttnResBertModel(config)
        self.head = LRAttnResBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         inputs_embeds=inputs_embeds,
                         output_hidden_states=output_hidden_states, return_dict=True)
        logits = self.decoder(self.head(out.last_hidden_state))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100)
        if not return_dict:
            rest = (logits,) + ((out.hidden_states,) if out.hidden_states is not None else ())
            return ((loss,) + rest) if loss is not None else rest
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=out.hidden_states)


class LRAttnResBertForSequenceClassification(LRAttnResBertPreTrainedModel):
    """Pooled (masked mean, or the first token) -> prediction head -> classifier."""

    def __init__(self, config: LRAttnResBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LRAttnResBertModel(config)
        self.head = LRAttnResBertPredictionHead(config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         inputs_embeds=inputs_embeds,
                         output_hidden_states=output_hidden_states, return_dict=True)
        h = out.last_hidden_state
        if self.config.classifier_pooling == "cls":
            pooled = h[:, 0]
        elif attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = attention_mask.to(h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.drop(self.head(pooled)))

        loss = None
        if labels is not None:
            problem = self.config.problem_type
            if problem is None:
                if self.num_labels == 1:
                    problem = "regression"
                elif labels.dtype in (torch.long, torch.int):
                    problem = "single_label_classification"
                else:
                    problem = "multi_label_classification"
                self.config.problem_type = problem
            if problem == "regression":
                loss = F.mse_loss(logits.squeeze(-1).float(), labels.float().view(-1)) \
                    if self.num_labels == 1 else F.mse_loss(logits.float(), labels.float())
            elif problem == "single_label_classification":
                loss = F.cross_entropy(logits.float().view(-1, self.num_labels), labels.view(-1))
            else:
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        if not return_dict:
            rest = (logits,) + ((out.hidden_states,) if out.hidden_states is not None else ())
            return ((loss,) + rest) if loss is not None else rest
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=out.hidden_states)


__all__ = [
    "LRAttnResBertPreTrainedModel",
    "LRAttnResBertModel",
    "LRAttnResBertForMaskedLM",
    "LRAttnResBertForSequenceClassification",
]
