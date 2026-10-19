"""CPU tests for LR-AttnRes-BERT.

Run with:  python -m pytest -q tests
"""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoModel, AutoModelForMaskedLM  # noqa: E402

from lr_attnres_bert import (  # noqa: E402
    LRAttnResBertConfig,
    LRAttnResBertForMaskedLM,
    LRAttnResBertForSequenceClassification,
    LRAttnResBertModel,
)
from lr_attnres_bert.modeling_lr_attnres_bert import (  # noqa: E402
    LRAttnResBertQuery,
    _group_ends,
)


def tiny(**kw):
    base = dict(vocab_size=128, hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
                num_key_value_heads=2, intermediate_size=128, attnres_rank=16,
                attnres_blocks=3, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                cls_token_id=1, sep_token_id=2, mask_token_id=3)
    base.update(kw)
    return LRAttnResBertConfig(**base)


def randomise_queries(model, std=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LRAttnResBertQuery):
                m.query.copy_(torch.randn(m.query.shape, generator=g) * std)


def batch(B=2, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 128, (B, T), generator=g)


# --- configuration -----------------------------------------------------------------

def test_default_is_modernbert_base_scale():
    cfg = LRAttnResBertConfig()
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads) == (22, 768, 12)
    assert cfg.num_key_value_heads == 4 and cfg.vocab_size == 50368
    assert (cfg.attnres_rank, cfg.attnres_blocks, cfg.attnres_read_sites) == (128, 8, "all")
    with torch.device("meta"):
        model = LRAttnResBertForMaskedLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 145e6 < n < 155e6                     # ModernBERT-base: 149.6 M
    assert model.decoder.weight is model.model.embeddings.weight


def test_invalid_configs_are_rejected():
    for bad in (dict(num_key_value_heads=3), dict(attnres_read_sites="mlp"),
                dict(attnres_rank=-1), dict(classifier_pooling="max")):
        with pytest.raises(ValueError):
            tiny(**bad)


def test_group_ends():
    # M = 2 * 22 = 44 writes in 8 groups end at writes ceil(44 j / 8)
    cfg = LRAttnResBertConfig()
    ends = [w for i in range(22) for k, w in ((0, 2 * i + 1), (1, 2 * i + 2))
            if _group_ends(cfg, i)[k]]
    assert ends == [6, 11, 17, 22, 28, 33, 39, 44]


def test_read_sites():
    # default "all" (the paper): every sublayer input and the output are routed
    model = LRAttnResBertModel(tiny())
    assert model.layers[0].res_attn is None              # sees the embedding alone
    assert all(layer.res_attn is not None for layer in model.layers[1:])
    assert all(layer.res_mlp is not None for layer in model.layers)
    assert model.res_out is not None
    att = LRAttnResBertModel(tiny(attnres_read_sites="attention"))
    assert all(layer.res_mlp is None for layer in att.layers)
    off = LRAttnResBertModel(tiny(attnres_rank=0))
    assert off.res_out is None and not any(n.endswith("query") for n, _ in off.named_parameters())


# --- the model ---------------------------------------------------------------------

def test_heads_run():
    ids = batch()
    mask = torch.ones_like(ids)
    mask[0, 9:] = 0
    lm = LRAttnResBertForMaskedLM(tiny()).eval()
    out = lm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    assert out.logits.shape == (2, 12, 128)
    assert len(out.hidden_states) == 1 + 6
    clf = LRAttnResBertForSequenceClassification(tiny(num_labels=3)).eval()
    out = clf(input_ids=ids, attention_mask=mask, labels=torch.tensor([0, 2]))
    assert out.logits.shape == (2, 3) and torch.isfinite(out.loss)


def test_zero_queries_read_the_plain_residual():
    """At init (zero queries) every routed read is the residual sum over the number of
    sources, and the RMSNorm after it removes that factor: LR-AttnRes-BERT computes
    what the plain encoder with the same weights computes. (Exactly so as the norm eps
    vanishes: with the init's 0.02-std embedding, a read divided by S sources has a
    mean square close to the default eps 1e-5, which then no longer cancels.)"""
    torch.manual_seed(0)
    on = LRAttnResBertModel(tiny(norm_eps=1e-12)).eval()
    off = LRAttnResBertModel(tiny(attnres_rank=0, norm_eps=1e-12)).eval()
    missing, unexpected = off.load_state_dict(on.state_dict(), strict=False)
    assert not missing and all(k.endswith("query") for k in unexpected)
    ids = batch()
    with torch.no_grad():
        a = on(input_ids=ids).last_hidden_state
        b = off(input_ids=ids).last_hidden_state
    assert F.cosine_similarity(a, b, dim=-1).min() > 1 - 1e-5
    randomise_queries(on)
    with torch.no_grad():
        c = on(input_ids=ids).last_hidden_state
    assert F.cosine_similarity(c, b, dim=-1).min() < 0.999


def test_routed_read_matches_a_reference():
    """The first routed read of layer 1 against a direct computation (attention-only
    read sites, so layer 0's MLP reads the plain sum)."""
    torch.manual_seed(0)
    model = LRAttnResBertModel(tiny(num_hidden_layers=2, attnres_blocks=0,
                                    attnres_read_sites="attention")).eval()
    randomise_queries(model)
    ids = batch()
    layer0, layer1 = model.layers
    seen = {}
    layer1.attn_norm.register_forward_hook(lambda m, i, o: seen.setdefault("in", i[0]))
    with torch.no_grad():
        model(input_ids=ids)
        x = model.embeddings(ids)
        cos, sin = model.rotary(ids.shape[1], x.device, x.dtype)
        a = layer0.attn(layer0.attn_norm(x), cos, sin)
        m = layer0.mlp(layer0.mlp_norm(x + a))     # MLP reads the plain sum
        srcs = [x, a, m]                           # attnres_blocks=0: one source per write
        r = model.attnres_rank
        keys = [F.rms_norm(s[..., -r:], (r,), eps=1e-6) for s in srcs]
        alpha = torch.stack([k @ layer1.res_attn.query for k in keys], -1).softmax(-1)
        ref = (alpha[..., None] * torch.stack(srcs, -2)).sum(-2)
    assert torch.allclose(seen["in"], ref, atol=1e-5)


@pytest.mark.parametrize("sites", ["attention", "all"])
def test_padding_does_not_reach_real_tokens(sites):
    torch.manual_seed(0)
    model = LRAttnResBertModel(tiny(attnres_read_sites=sites)).double().eval()
    randomise_queries(model)
    ids = batch(B=2, T=16)
    mask = torch.ones_like(ids)
    mask[1, 9:] = 0
    with torch.no_grad():
        padded = model(input_ids=ids, attention_mask=mask).last_hidden_state
        alone = model(input_ids=ids[1:, :9]).last_hidden_state
    assert torch.allclose(padded[1, :9], alone[0], atol=1e-10)


def test_all_parameters_get_gradients():
    model = LRAttnResBertForMaskedLM(tiny()).train()
    ids = batch()
    labels = ids.clone()
    labels[:, ::2] = -100
    model(input_ids=ids, labels=labels).loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert dead == []


def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = LRAttnResBertForMaskedLM(tiny()).eval()
    randomise_queries(model)
    ids = batch()
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    model.save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["model_type"] == "lr_attnres_bert"
    assert cfg["auto_map"]["AutoModelForMaskedLM"].endswith("LRAttnResBertForMaskedLM")
    assert (tmp_path / "modeling_lr_attnres_bert.py").exists()
    for cls in (LRAttnResBertForMaskedLM, AutoModelForMaskedLM):
        loaded = cls.from_pretrained(tmp_path).eval()
        with torch.no_grad():
            assert torch.equal(loaded(input_ids=ids).logits, ref)
    # the bare encoder loads from a masked-LM checkpoint
    enc = AutoModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        assert torch.equal(enc(input_ids=ids).last_hidden_state,
                           model.model(input_ids=ids).last_hidden_state)


def test_bf16_inference_runs():
    model = LRAttnResBertModel(tiny()).to(torch.bfloat16).eval()
    randomise_queries(model, std=0.1)
    with torch.inference_mode():
        out = model(input_ids=batch()).last_hidden_state
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


def test_tiny_preset():
    cfg = LRAttnResBertConfig.from_preset("tiny")
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.intermediate_size, cfg.attnres_rank) == (
        8, 384, 6, 2, 1152, 64)
    # the base ratios: MLP 3x, GQA 3:1, head dim 64, rank = hidden / 6
    base = LRAttnResBertConfig.from_preset("base")
    for c in (cfg, base):
        assert c.intermediate_size == 3 * c.hidden_size
        assert c.hidden_size // c.num_attention_heads == 64
        assert c.attnres_rank * 6 == c.hidden_size
    with torch.device("meta"):
        n = sum(p.numel() for p in LRAttnResBertForMaskedLM(cfg).parameters())
    assert 28e6 < n < 32e6
    assert LRAttnResBertConfig.from_preset("tiny", attnres_rank=0).attnres_rank == 0
    with pytest.raises(ValueError):
        LRAttnResBertConfig.from_preset("huge")


def test_autocast_keeps_the_stream_in_full_precision():
    """Mixed-precision fine-tuning: group sums stay in the embedding's dtype and the
    routing runs outside autocast. (Forward only: CPU oneDNN has no bf16 backward
    on every machine.)"""
    from lr_attnres_bert.modeling_lr_attnres_bert import _Stream

    torch.manual_seed(0)
    model = LRAttnResBertForMaskedLM(tiny()).train()
    randomise_queries(model, std=0.3)
    dtypes = []
    orig = _Stream.write

    def spy(self, o, close, w):
        orig(self, o, close, w)
        dtypes.extend({s.dtype for s in self.sources} | {self.total.dtype})
    _Stream.write = spy
    try:
        ids = batch()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
    finally:
        _Stream.write = orig
    assert set(dtypes) == {torch.float32}
    assert torch.isfinite(loss) and loss.requires_grad
