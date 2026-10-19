#!/usr/bin/env python
"""Turn a benchmark_inference.py JSON into the markdown tables of the README.

    python benchmarks/make_tables.py benchmarks/results/rtx5080_bf16.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABELS = {"lr-attnres-bert": "LR-AttnRes-BERT", "no-attnres": "same, LR-AttnRes off",
          "modernbert-base": "ModernBERT-base"}


def main(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    env, rows = data["env"], data["results"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    shapes = list(dict.fromkeys((r["batch"], r["length"]) for r in rows))
    cell = {(r["model"], r["mode"], r["batch"], r["length"]): r for r in rows}

    def get(m, mode, b, t):
        r = cell.get((m, mode, b, t))
        return None if r is None or r.get("oom") else r

    out = [f"_{env['device']}, torch {env['torch']} (CUDA {env['cuda']}), transformers "
           f"{env['transformers']}, {env['dtype']}, median of {env['iters']} forwards "
           f"after {env['warmup']} warm-up._", ""]
    head = "| batch x length | " + " | ".join(LABELS.get(m, m) for m in models)
    for mode in modes:
        out += [f"**Latency (ms) and throughput (k tokens/s), {mode}**", "",
                head + " | LR-AttnRes overhead |",
                "|---|" + "---:|" * (len(models) + 1)]
        for b, t in shapes:
            cells = []
            for m in models:
                r = get(m, mode, b, t)
                cells.append("OOM" if r is None else
                             f"{r['latency_ms']:.2f} ({r['tokens_per_s'] / 1e3:.0f}k)")
            on, off = get("lr-attnres-bert", mode, b, t), get("no-attnres", mode, b, t)
            ovh = f"{on['latency_ms'] / off['latency_ms'] - 1:+.1%}" if on and off else "--"
            out.append(f"| {b} x {t} | " + " | ".join(cells) + f" | {ovh} |")
        out.append("")
    for mode in modes:
        out += [f"**Peak memory (GiB, weights included), {mode}**", "", head + " |",
                "|---|" + "---:|" * len(models)]
        w = []
        for m in models:
            r = next((x for x in rows if x["model"] == m and x["mode"] == mode), None)
            w.append("--" if r is None or r.get("weights_gib") is None else f"{r['weights_gib']:.3f}")
        out.append("| weights only | " + " | ".join(w) + " |")
        for b, t in shapes:
            cells = []
            for m in models:
                r = get(m, mode, b, t)
                cells.append("OOM" if r is None else f"{r['peak_gib']:.3f}")
            out.append(f"| {b} x {t} | " + " | ".join(cells) + " |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/rtx5080_bf16.json")
