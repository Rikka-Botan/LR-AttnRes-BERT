#!/usr/bin/env python
"""LR-AttnRes-BERT quickstart with the local package: fill-mask, embeddings, classifier.

    python examples/quickstart.py                       # RikkaBotan/LR-AttnRes-BERT-tiny
    python examples/quickstart.py --model <checkpoint dir or Hub id>
    python examples/quickstart.py --device cuda --compile

Unlike ``fill_mask.py`` / ``embeddings.py`` (``trust_remote_code``), this imports the
classes from ``src/lr_attnres_bert``. ``--random`` builds an untrained model instead,
which shows the API but predicts noise.
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
    p.add_argument("--model", default="RikkaBotan/LR-AttnRes-BERT-tiny",
                   help="checkpoint directory or Hub id")
    p.add_argument("--random", action="store_true", help="untrained tiny model (API demo)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="torch.compile the encoder")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # LR-AttnRes-BERT uses ModernBERT's tokenizer (same vocabulary and special tokens)
    tok = AutoTokenizer.from_pretrained(
        "answerdotai/ModernBERT-base" if args.random else args.model)
    if args.random:
        print("[quickstart] --random: untrained weights, the predictions below are noise")
        mlm = LRAttnResBertForMaskedLM(LRAttnResBertConfig.from_preset("tiny"))
    else:
        mlm = LRAttnResBertForMaskedLM.from_pretrained(args.model)
    mlm = mlm.to(device=device, dtype=dtype).eval()
    run = torch.compile(mlm) if args.compile else mlm

    # 1. fill-mask
    text = f"The capital of France is {tok.mask_token}."
    enc = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = run(**enc).logits
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    top = logits[0, pos].float().softmax(-1).topk(5)
    print(text)
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {tok.decode([idx]).strip()!r:>12}  {prob:.3f}")

    # 2. sentence embeddings: masked mean of the encoder output
    sentences = ["A man is playing a guitar.",
                 "Someone is playing an instrument.",
                 "The stock market fell sharply today."]
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
        out = clf(**enc).logits.float().cpu()
    print("classifier logits (untrained head):", out.shape, out.numpy().round(3).tolist()[0])


if __name__ == "__main__":
    main()
