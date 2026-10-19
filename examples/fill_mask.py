#!/usr/bin/env python
"""Fill-mask with a pretrained LR-AttnRes-BERT from the Hugging Face Hub.

Needs only ``transformers`` -- the modeling code is fetched with the weights
(``trust_remote_code=True``).

    python examples/fill_mask.py
    python examples/fill_mask.py --text "Paris is the [MASK] of France." --top_k 10
    python examples/fill_mask.py --device cuda --dtype bf16
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
EXAMPLES = [
    "The capital of France is [MASK].",
    "I like to drink [MASK] in the morning.",
    "The cat sat on the [MASK].",
    "def add(a, b):\n    return a [MASK] b",
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="RikkaBotan/LR-AttnRes-BERT-tiny")
    p.add_argument("--text", action="append", help="sentence(s) with [MASK]; default: examples")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    args = p.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=DTYPES[args.dtype]).to(args.device).eval()

    texts = [t.replace("[MASK]", tok.mask_token) for t in (args.text or EXAMPLES)]
    batch = tok(texts, padding=True, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        logits = model(**batch).logits

    for i, text in enumerate(texts):
        print(text.replace("\n", "\\n"))
        for pos in (batch["input_ids"][i] == tok.mask_token_id).nonzero().flatten().tolist():
            probs = logits[i, pos].float().softmax(-1)
            top = probs.topk(args.top_k)
            print("   " + "  ".join(f"{tok.decode([t]).strip()!r} {p:.3f}"
                                    for p, t in zip(top.values.tolist(), top.indices.tolist())))


if __name__ == "__main__":
    main()
