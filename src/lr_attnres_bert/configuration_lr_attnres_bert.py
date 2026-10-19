"""LR-AttnRes-BERT configuration."""

from __future__ import annotations

try:
    from transformers import PretrainedConfig
except ImportError:  # pragma: no cover - newer alias
    from transformers import PreTrainedConfig as PretrainedConfig


ATTNRES_READ_SITES = ("attention", "all")
CLASSIFIER_POOLINGS = ("mean", "cls")

# Named sizes (overrides on top of the defaults, which are "base").
#   base  22 layers x 768, 12 / 4 heads, MLP 2304, rank 128   ~151.8 M (ModernBERT-base scale)
#   tiny   8 layers x 384,  6 / 2 heads, MLP 1152, rank  64    ~29.9 M (same ratios: MLP 3x,
#                                                                        GQA 3:1, head dim 64,
#                                                                        rank = hidden / 6)
PRESETS = {
    "base": {},
    "tiny": dict(hidden_size=384, num_hidden_layers=8, num_attention_heads=6,
                 num_key_value_heads=2, intermediate_size=1152, attnres_rank=64),
}


class LRAttnResBertConfig(PretrainedConfig):
    """Configuration of an LR-AttnRes-BERT encoder.

    The defaults are the base model: ModernBERT-base's depth and width (22 layers,
    hidden 768, 12 attention heads, vocabulary 50368 with ModernBERT's tokenizer and
    special-token ids), with a plain pre-norm encoder layer -- grouped-query
    attention with rotary positions and a squared-ReLU MLP -- and Block Sliced
    Low-Rank Attention Residuals in place of the residual sum.

    LR-AttnRes (``attnres_*``):

    * ``attnres_rank`` -- the routing key is the RMS-normalised last ``rank``
      channels of each residual source; 0 disables LR-AttnRes (a plain residual
      encoder with otherwise identical weights).
    * ``attnres_blocks`` -- the ``2 * num_hidden_layers`` sublayer writes are summed
      into this many groups (Block AttnRes; 8 = the paper's best at rank 128); 0
      keeps every write as its own source.
    * ``attnres_read_sites`` -- ``"all"`` (the paper): every sublayer input and the
      encoder output are routed; ``"attention"``: only the attention inputs and the
      output, the MLPs reading the plain residual sum (cheaper, not the paper's).
    """

    model_type = "lr_attnres_bert"

    def __init__(
        self,
        vocab_size: int = 50368,
        hidden_size: int = 768,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 2304,
        max_position_embeddings: int = 8192,
        rope_theta: float = 160000.0,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        attnres_rank: int = 128,
        attnres_blocks: int = 8,
        attnres_read_sites: str = "all",
        classifier_pooling: str = "mean",
        classifier_dropout: float = 0.0,
        pad_token_id: int = 50283,
        bos_token_id: int = 50281,
        eos_token_id: int = 50282,
        cls_token_id: int = 50281,
        sep_token_id: int = 50282,
        mask_token_id: int = 50284,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if hidden_size % num_attention_heads:
            raise ValueError(f"hidden_size {hidden_size} is not a multiple of "
                             f"num_attention_heads {num_attention_heads}")
        if num_attention_heads % num_key_value_heads:
            raise ValueError(f"num_attention_heads {num_attention_heads} is not a multiple "
                             f"of num_key_value_heads {num_key_value_heads}")
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("the head dimension must be even for rotary embeddings")
        if attnres_rank < 0 or attnres_blocks < 0:
            raise ValueError("attnres_rank and attnres_blocks must be >= 0")
        if attnres_read_sites not in ATTNRES_READ_SITES:
            raise ValueError(f"attnres_read_sites must be one of {ATTNRES_READ_SITES}, "
                             f"got {attnres_read_sites!r}")
        if classifier_pooling not in CLASSIFIER_POOLINGS:
            raise ValueError(f"classifier_pooling must be one of {CLASSIFIER_POOLINGS}, "
                             f"got {classifier_pooling!r}")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = float(rope_theta)
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.attnres_rank = int(attnres_rank)
        self.attnres_blocks = int(attnres_blocks)
        self.attnres_read_sites = attnres_read_sites
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "LRAttnResBertConfig":
        """A named size (``PRESETS``: "base", "tiny") with optional overrides."""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose one of {tuple(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})


__all__ = ["LRAttnResBertConfig", "ATTNRES_READ_SITES", "CLASSIFIER_POOLINGS", "PRESETS"]
