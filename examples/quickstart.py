#!/usr/bin/env python
"""LR-AttnRes-BERT quickstart: fill-mask, sentence embeddings and the classification head.

    python examples/quickstart.py --model <checkpoint dir or Hub id>

LR-AttnRes-BERT uses ModernBERT's tokenizer (same vocabulary and special tokens).
Without ``--model`` a randomly initialised model is built, which shows the API but
predicts nonsense -- no pretrained weights ship with this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from lr_attnres_bert import (  # noqa: E402
    LRAttnResBertConfig,
    LRAttnResBertForMaskedLM,
    LRAttnResBertForSequenceClassification,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="checkpoint directory or Hub id")
    p.add_argument("--tokenizer", default="answerdotai/ModernBERT-base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.model:
        mlm = LRAttnResBertForMaskedLM.from_pretrained(args.model, dtype=dtype)
    else:
        print("[quickstart] no --model: random weights, the predictions below are noise")
        mlm = LRAttnResBertForMaskedLM(LRAttnResBertConfig()).to(dtype)
    mlm = mlm.to(device).eval()

    # 1. fill-mask
    text = f"The capital of France is {tok.mask_token}."
    enc = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = mlm(**enc).logits
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    top = logits[0, pos].float().softmax(-1).topk(5)
    print(text)
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {tok.decode([idx])!r:>14}  {prob:.3f}")

    # 2. sentence embeddings: masked mean of the encoder output
    sentences = ["LR-AttnRes routes each read over depth.",
                 "Attention residuals mix earlier layers' outputs.",
                 "Paris is the capital of France."]
    enc = tok(sentences, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        h = mlm.model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    emb = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)
    print("cosine similarity:\n", (emb @ emb.T).cpu().numpy().round(3))

    # 3. a sequence-classification head on the same encoder (fine-tune before use)
    clf = LRAttnResBertForSequenceClassification(
        LRAttnResBertConfig(**{**mlm.config.to_dict(), "num_labels": 2}))
    clf.model.load_state_dict(mlm.model.state_dict())
    clf = clf.to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        print("classifier logits:", clf(**enc).logits.float().cpu().numpy().round(3))


if __name__ == "__main__":
    main()
