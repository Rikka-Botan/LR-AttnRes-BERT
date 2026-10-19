#!/usr/bin/env python
"""Figures and tables for the LR-AttnRes ablation of LR-AttnRes-BERT Tiny.

Reads the two scoring JSONs in ``results/tiny_ablation/`` (pseudo-perplexity and
MLM accuracy per sequence length, each the mean and 95% Student-t CI over 20
disjoint measurements) and writes

* ``docs/figures/fig1_quality_vs_length.{png,pdf}`` -- PPPL and MLM accuracy vs length,
* ``docs/figures/fig2_difference_vs_length.{png,pdf}`` -- LR-AttnRes minus plain,
* the markdown tables of the README to stdout (``--tables``).

    python results/make_figures.py            # figures
    python results/make_figures.py --tables   # + tables
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tiny_ablation"
FIGURES = ROOT / "docs" / "figures"
TRAINED_LEN = 1024

# Categorical slots 1 and 2 of the reference palette (validated light mode:
# CVD dE 24.7, normal-vision dE 33.6); marker shape is the print/CVD backup.
ARMS = [
    ("tiny_attnres", "LR-AttnRes", "#2a78d6", "o"),
    ("tiny_plain", "Plain residual", "#eb6834", "s"),
]
INK, INK_2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"


def load():
    data = {}
    for key, *_ in ARMS:
        d = json.loads((RESULTS / f"{key}.json").read_text(encoding="utf-8"))
        rows = {int(L): r for L, r in d["results"].items() if r}
        data[key] = {"meta": d, "rows": rows}
    lengths = sorted(set.intersection(*(set(v["rows"]) for v in data.values())))
    return data, lengths


def series(data, key, metric, lengths):
    rows = data[key]["rows"]
    scale = 100.0 if metric == "mlm_acc" else 1.0
    return ([rows[L][metric] * scale for L in lengths],
            [rows[L][metric + "_ci95"] * scale for L in lengths])


def style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def frame(ax, lengths):
    ax.set_xscale("log", base=2)
    ax.set_xlim(lengths[0] / 1.4, lengths[-1] * 1.4)
    ax.set_xticks([2 ** k for k in range(3, 17, 2)])
    ax.set_xticklabels([f"$2^{{{k}}}$" for k in range(3, 17, 2)])
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    # the lengths beyond training: shaded, labelled once per panel
    ax.axvspan(TRAINED_LEN * 1.0, lengths[-1] * 1.4, color=BAND, zorder=0, lw=0)
    ax.set_xlabel("Sequence length (tokens)")


def panel_label(ax, text):
    ax.text(-0.13, 1.04, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def figure1(plt, data, lengths):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), constrained_layout=True)
    for ax, metric, ylabel in ((axes[0], "pppl", "Pseudo-perplexity (log scale)"),
                               (axes[1], "mlm_acc", "MLM accuracy (%)")):
        frame(ax, lengths)
        for key, label, color, marker in ARMS:
            y, ci = series(data, key, metric, lengths)
            ax.errorbar(lengths, y, yerr=ci, color=color, marker=marker, markersize=3.6,
                        linewidth=1.3, elinewidth=0.9, capsize=1.8, label=label,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_ylabel(ylabel)
        if metric == "pppl":
            ax.set_yscale("log")
            ax.set_ylim(10, 8000)
            ax.set_yticks([10, 30, 100, 300, 1000, 3000])
            ax.set_yticklabels(["10", "30", "100", "300", "1,000", "3,000"])
            ax.minorticks_off()
            ax.tick_params(axis="x", which="minor", bottom=False)
        else:
            ax.set_ylim(0, 56)
        # the band label sits where no curve passes: bottom right in (a), mid right in (b)
        ax.text(0.985, 0.04 if metric == "pppl" else 0.62, "beyond\ntraining length",
                transform=ax.transAxes, fontsize=7, color=INK_2, va="bottom", ha="right",
                linespacing=1.1)
    axes[0].legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.2,
                   bbox_to_anchor=(0.06, 1.0))
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def difference(data, metric, lengths):
    a, ca = series(data, "tiny_attnres", metric, lengths)
    p, cp = series(data, "tiny_plain", metric, lengths)
    if metric == "pppl":   # relative change, %, CI by the delta method
        d = [100.0 * (x / y - 1.0) for x, y in zip(a, p)]
        e = [100.0 * (x / y) * math.hypot(cx / x, cy / y) for x, y, cx, cy in zip(a, p, ca, cp)]
    else:                  # accuracy points
        d = [x - y for x, y in zip(a, p)]
        e = [math.hypot(cx, cy) for cx, cy in zip(ca, cp)]
    return d, e


def figure2(plt, data, lengths):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.45), constrained_layout=True)
    color, marker = ARMS[0][2], ARMS[0][3]
    for ax, metric, ylabel, lim in (
            (axes[0], "pppl", "Δ pseudo-perplexity vs plain (%)", (-60, 100)),
            (axes[1], "mlm_acc", "Δ MLM accuracy vs plain (points)", (-5, 6))):
        frame(ax, lengths)
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
        d, e = difference(data, metric, lengths)
        ax.errorbar(lengths, d, yerr=e, color=color, marker=marker, markersize=3.6,
                    linewidth=1.3, elinewidth=0.9, capsize=1.8, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3)
        ax.set_ylim(*lim)
        ax.set_ylabel(ylabel)
        better = "lower is better" if metric == "pppl" else "higher is better"
        ax.text(0.02, 0.97, better, transform=ax.transAxes, fontsize=7, color=INK_2,
                va="top", ha="left")
        # band label where no curve passes: bottom right in (a), top right in (b)
        top = metric != "pppl"
        ax.text(0.985, 0.97 if top else 0.04, "beyond\ntraining length", transform=ax.transAxes,
                fontsize=7, color=INK_2, va="top" if top else "bottom", ha="right", linespacing=1.1)
        # points clipped by the axis range are marked with their value
        lo, hi = lim
        for L, v in zip(lengths, d):
            if v > hi or v < lo:
                ax.annotate(f"{v:+.0f}%", xy=(L, hi if v > hi else lo), xytext=(0, -9 if v > hi else 3),
                            textcoords="offset points", ha="center", fontsize=6.5, color=INK_2)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def fmt(v, ci, digits):
    return f"{v:.{digits}f} ± {ci:.{digits}f}"


def tables(data, lengths):
    show = [8, 32, 128, 512, 1024, 2048, 4096, 8192]
    out = []
    for metric, title, digits, better in (("pppl", "Pseudo-perplexity ↓", 2, min),
                                          ("mlm_acc", "MLM accuracy (%) ↑", 2, max)):
        cols = [L for L in show if L in lengths]
        head = " | ".join(f"{L:,}" + ("†" if L > TRAINED_LEN else "") for L in cols)
        out += [f"**{title}**", "", f"| Model | {head} |",
                "|:--|" + "--:|" * len(cols)]
        vals = {k: series(data, k, metric, cols) for k, *_ in ARMS}
        for key, label, *_ in ARMS:
            cells = []
            for i, L in enumerate(cols):
                v, ci = vals[key][0][i], vals[key][1][i]
                other = [k for k, *_ in ARMS if k != key][0]
                ov, oci = vals[other][0][i], vals[other][1][i]
                disjoint = (v + ci < ov - oci) or (v - ci > ov + oci)
                is_better = better(v, ov) == v and v != ov
                text = fmt(v, ci, digits)
                cells.append(f"**{text}**" if disjoint and is_better else text)
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    # appendix: every length
    out += ["<details><summary>All 14 lengths</summary>", ""]
    for metric, title, digits in (("pppl", "Pseudo-perplexity", 3), ("mlm_acc", "MLM accuracy (%)", 2)):
        out += [f"**{title}**", "", "| Length | LR-AttnRes | Plain residual |", "|--:|--:|--:|"]
        a = series(data, "tiny_attnres", metric, lengths)
        p = series(data, "tiny_plain", metric, lengths)
        for i, L in enumerate(lengths):
            out.append(f"| {L:,} | {fmt(a[0][i], a[1][i], digits)} | {fmt(p[0][i], p[1][i], digits)} |")
        out.append("")
    out += ["</details>", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true", help="also print the markdown tables")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):   # the tables carry ± ↓ † on any console
        sys.stdout.reconfigure(encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    data, lengths = load()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, make in (("fig1_quality_vs_length", figure1), ("fig2_difference_vs_length", figure2)):
        fig = make(plt, data, lengths)
        for ext in ("png", "pdf"):
            fig.savefig(FIGURES / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"wrote {FIGURES / name}.png / .pdf")
    if args.tables:
        print(tables(data, lengths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
