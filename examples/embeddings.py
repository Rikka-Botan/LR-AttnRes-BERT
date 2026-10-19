#!/usr/bin/env python
"""Sentence embeddings from a pretrained LR-AttnRes-BERT (masked mean pooling).

The encoder is a masked-LM backbone, not a trained sentence encoder: the cosine
similarities below are those of raw MLM features. Fine-tune (e.g. contrastively)
before using it for retrieval.

    python examples/embeddings.py
    python examples/embeddings.py --device cuda --dtype bf16
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
SENTENCES = [
    "A man is playing a guitar.",
    "Someone is playing an instrument.",
    "The stock market fell sharply today.",
    "Share prices dropped on Wall Street.",
]


def embed(model, tok, sentences, device):
    batch = tok(sentences, padding=True, truncation=True, max_length=8192,
                return_tensors="pt").to(device)
    with torch.inference_mode():
        h = model(**batch).last_hidden_state                      # (B, T, H)
    m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="RikkaBotan/LR-AttnRes-BERT-tiny")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    args = p.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    # the checkpoint is a masked LM: AutoModel loads its encoder and would list the
    # unused MLM-head weights as UNEXPECTED -- expected, so the report is muted
    transformers.logging.set_verbosity_error()
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=DTYPES[args.dtype]).to(args.device).eval()
    emb = embed(model, tok, SENTENCES, args.device)
    print("embedding shape:", tuple(emb.shape))
    sim = (emb @ emb.T).cpu()
    for i, s in enumerate(SENTENCES):
        print(f"{s:40s} " + " ".join(f"{v:6.3f}" for v in sim[i].tolist()))


if __name__ == "__main__":
    main()
