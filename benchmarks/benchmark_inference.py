#!/usr/bin/env python
"""Inference speed and memory of LR-AttnRes-BERT.

Times the encoder forward pass (``last_hidden_state``; no vocabulary projection)
under ``torch.inference_mode`` for every (model, mode, batch x length) cell and
records the median latency, the throughput and the CUDA memory:

* ``weights``  -- allocated by loading the model, before any forward;
* ``peak``     -- the maximum allocated during the timed forwards, weights included;
  ``activations`` = peak - weights.

Both are measured against the allocator's state right before the model was built,
after collecting the previous model, so the cells do not leak into each other.

Models:

* ``lr-attnres-bert``  this repository's default configuration (random weights --
  speed and memory do not depend on their values);
* ``no-attnres``       the same configuration with ``attnres_rank=0``: the plain
  residual encoder, i.e. exactly what LR-AttnRes costs;
* ``modernbert-base``  ``answerdotai/ModernBERT-base`` through ``transformers``
  (SDPA attention), the reference at the same depth and width.

Every input is full-length (no padding), so no model takes a masked path.

Usage:
    python benchmarks/benchmark_inference.py                       # CUDA, bf16
    python benchmarks/benchmark_inference.py --modes eager --shapes 8x512
    python benchmarks/benchmark_inference.py --device cpu --dtype fp32 --shapes 1x128,8x512
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import transformers  # noqa: E402

from lr_attnres_bert import LRAttnResBertConfig, LRAttnResBertModel  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODELS = ("lr-attnres-bert", "no-attnres", "modernbert-base")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--modes", default="eager,compile", help="eager and/or compile")
    p.add_argument("--shapes", default="64x128,32x512,8x2048,2x8192",
                   help="comma-separated BATCHxLENGTH")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--modernbert", default="answerdotai/ModernBERT-base")
    p.add_argument("--out", default=None, help="results JSON (default: benchmarks/results/<device>.json)")
    return p.parse_args(argv)


def build(name: str, args, device, dtype):
    if name == "modernbert-base":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(args.modernbert, dtype=dtype,
                                          attn_implementation="sdpa")
    else:
        cfg = LRAttnResBertConfig(attnres_rank=0 if name == "no-attnres" else 128)
        torch.manual_seed(0)
        model = LRAttnResBertModel(cfg).to(dtype)
        if name == "lr-attnres-bert":
            # Non-zero queries, as a trained model has (the cost is the same either
            # way; this just keeps the measured path the routed one).
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n.endswith("query"):
                        p.normal_(std=0.1)
    return model.to(device).eval()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_cell(model, B, T, args, device, vocab, base):
    ids = torch.randint(5, vocab - 100, (B, T), device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_ids=ids)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(input_ids=ids)
            sync(device)
            times.append(time.perf_counter() - t0)
    ms = statistics.median(times) * 1000
    row = {"batch": B, "length": T, "latency_ms": round(ms, 3),
           "tokens_per_s": round(B * T / (ms / 1000)), "ms_min": round(min(times) * 1000, 3)}
    if device.type == "cuda":
        row["peak_gib"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 3)
    return row


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    shapes = [tuple(int(v) for v in s.lower().split("x")) for s in args.shapes.split(",")]
    env = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    print(json.dumps(env))
    results = []
    for name in args.models.split(","):
        for mode in args.modes.split(","):
            gc.collect()
            base = 0
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            model = build(name, args, device, dtype)
            n_params = sum(p.numel() for p in model.parameters())
            weights = ((torch.cuda.memory_allocated() - base) / 2 ** 30
                       if device.type == "cuda" else None)
            vocab = model.get_input_embeddings().num_embeddings
            run = torch.compile(model) if mode == "compile" else model
            for B, T in shapes:
                try:
                    row = time_cell(run, B, T, args, device, vocab, base)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row = {"batch": B, "length": T, "oom": True}
                row.update(model=name, mode=mode, params=n_params,
                           weights_gib=None if weights is None else round(weights, 3))
                if "peak_gib" in row and weights is not None:
                    row["activations_gib"] = round(row["peak_gib"] - weights, 3)
                results.append(row)
                print(json.dumps(row), flush=True)
            del run, model
            gc.collect()
            torch._dynamo.reset()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = Path(args.out or ROOT / "benchmarks" / "results" / f"{device.type}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
