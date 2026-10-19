"""LR-AttnRes-BERT: a GQA + RoPE + ReLU^2 encoder with Block Sliced Low-Rank
Attention Residuals (inference code).

Importing the package registers the classes with the ``transformers`` Auto
factories, and marks them for ``register_for_auto_class`` so ``save_pretrained``
writes an ``auto_map`` and copies this code next to the weights (load such a
directory or Hub repo with ``trust_remote_code=True``).
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

from .configuration_lr_attnres_bert import LRAttnResBertConfig
from .modeling_lr_attnres_bert import (
    LRAttnResBertForMaskedLM,
    LRAttnResBertForSequenceClassification,
    LRAttnResBertModel,
    LRAttnResBertPreTrainedModel,
)

__version__ = "0.1.0"


def _register():
    try:
        AutoConfig.register(LRAttnResBertConfig.model_type, LRAttnResBertConfig)
    except ValueError:  # already registered (re-import)
        return
    AutoModel.register(LRAttnResBertConfig, LRAttnResBertModel)
    AutoModelForMaskedLM.register(LRAttnResBertConfig, LRAttnResBertForMaskedLM)
    AutoModelForSequenceClassification.register(
        LRAttnResBertConfig, LRAttnResBertForSequenceClassification)
    LRAttnResBertConfig.register_for_auto_class()
    LRAttnResBertModel.register_for_auto_class("AutoModel")
    LRAttnResBertForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
    LRAttnResBertForSequenceClassification.register_for_auto_class(
        "AutoModelForSequenceClassification")


_register()

__all__ = [
    "LRAttnResBertConfig",
    "LRAttnResBertPreTrainedModel",
    "LRAttnResBertModel",
    "LRAttnResBertForMaskedLM",
    "LRAttnResBertForSequenceClassification",
    "__version__",
]
