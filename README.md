# LR-AttnRes-BERT

A BERT-style masked-language-model encoder at **ModernBERT-base scale** (22 layers,
hidden 768, 151.8 M parameters) built from deliberately plain layers —
**grouped-query attention + RoPE + a squared-ReLU MLP** — with **Block Sliced
Low-Rank Attention Residuals (LR-AttnRes)** in place of the residual sum.

This repository is **inference code only** (PyTorch + 🤗 `transformers`): the model,
masked-LM and sequence-classification heads, `save_pretrained` / `from_pretrained`
support, and the speed / memory measurements below. There is no training code and
**no pretrained checkpoint is published yet** — a model built from the config has
random weights.

## Architecture

| | LR-AttnRes-BERT | ModernBERT-base (reference) |
|---|---|---|
| layers / hidden | 22 / 768 | 22 / 768 |
| attention | 12 query heads, **4 KV heads (GQA)**, head dim 64, full (global) attention in every layer | 12 heads (MHA); global every 3rd layer, 128-token sliding window otherwise |
| positions | RoPE, θ = 160 000 | RoPE, θ = 160 000 (global) / 10 000 (local) |
| MLP | **ReLU²**: `W_o relu(W_i x)²`, 768 → 2304 → 768 | GeGLU, 768 → 2 × 1152 → 768 |
| norm | RMSNorm, pre-norm | LayerNorm (no bias), pre-norm |
| residual | **Block Sliced LR-AttnRes** | sum |
| bias terms | none in the encoder | none in the encoder |
| vocabulary / tokenizer | 50 368, ModernBERT's tokenizer and special-token ids | same |
| max length | 8192 | 8192 |
| parameters (MLM / encoder) | 151.8 M / 151.2 M | 149.6 M / 149.0 M |

The MLP width (2304 = 3 × 768) is chosen so that the non-gated ReLU² MLP plus the
cheaper GQA attention land at ModernBERT-base's parameter count (+1.5%).

### LR-AttnRes

In a standard pre-norm encoder every sublayer reads the residual sum
`x = emb + Σ writes`. Attention Residuals replace a read by a softmax mix over depth
of the earlier residual writes; the **sliced low-rank** variant (Su, 2026) routes on
only the last `r` channels of each source:

```
alpha_i = softmax_i( q_t · RMSNorm(tail_r(s_i)) )      q_t ∈ R^r, learned, one per read
h_t     = Σ_i alpha_i s_i                              (full-width values)
```

* **Sources (Block AttnRes).** The 44 sublayer writes (22 attention + 22 MLP) are
  summed into `attnres_blocks = 8` groups, closing after writes 6, 11, 17, 22, 28,
  33, 39 and 44. The sources of a read are the embedding, the closed groups, and the
  open group's partial sum — at most nine.
* **Routed reads** (`attnres_read_sites = "all"`, as in the paper): every attention
  input, every MLP input and the encoder output read the mix. The first attention
  input sees only the embedding and has no query, so there are 44 queries of rank
  `r = 128`: **5 632 parameters** in all. `"attention"` routes only the attention
  inputs and the output and lets the MLPs read the plain residual sum — cheaper, but
  not the paper's setup.
* **Initialisation.** The queries start at zero: every routed read is then the
  residual sum divided by the number of sources, and the RMSNorm that follows removes
  that factor — a fresh LR-AttnRes-BERT computes what the plain encoder with the same
  weights computes. `attnres_rank = 0` *is* that plain encoder (no queries).
* **Evaluation.** Between two group closes the closed sources and the queries are
  fixed, so a close mixes the closed sources for every read of the coming segment
  in one batched matmul (keeping their log-sum-exp), and each read merges in the
  open partial sum with one online-softmax step,
  `h = lerp(hc, partial, sigmoid(q · key(partial) − lse))`. The result is exactly the
  softmax above; it just touches the closed sources once per segment instead of once
  per read.

### Presets

| preset | layers × hidden | heads (query / KV) | MLP | LR-AttnRes rank | params |
|---|---|---|---|---|---:|
| `base` (default) | 22 × 768 | 12 / 4 | 2304 | 128 | 151.8 M |
| `tiny` | 8 × 384 | 6 / 2 | 1152 | 64 | 29.9 M |

`tiny` keeps the base ratios (MLP 3×, GQA 3:1, head dim 64, rank = hidden / 6):
`LRAttnResBertConfig.from_preset("tiny")` — a ~30 M model for quick ablations (with
and without LR-AttnRes). This repository itself is inference-only; the modules are
plain PyTorch, so any MLM training loop can train them (mixed precision included: under
autocast the residual sources stay in fp32 and the routing runs outside autocast).

## Install

```bash
pip install -e .            # torch >= 2.4, transformers >= 5.0
```

The tokenizer is ModernBERT's: `AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")`.

## Usage

```python
import torch
from transformers import AutoTokenizer
from lr_attnres_bert import LRAttnResBertConfig, LRAttnResBertForMaskedLM

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = LRAttnResBertForMaskedLM.from_pretrained("path/to/checkpoint", dtype=torch.bfloat16)
# or, untrained: model = LRAttnResBertForMaskedLM(LRAttnResBertConfig())
model = model.cuda().eval()

enc = tok(f"The capital of France is {tok.mask_token}.", return_tensors="pt").to("cuda")
with torch.inference_mode():
    logits = model(**enc).logits                       # (1, T, 50368)
    hidden = model.model(**enc).last_hidden_state      # (1, T, 768) encoder output

fast = torch.compile(model)                            # see the benchmarks
```

* `LRAttnResBertModel` — the encoder (`last_hidden_state`; `output_hidden_states`
  returns the embedding and the plain residual sum after each layer);
  `LRAttnResBertForMaskedLM`; `LRAttnResBertForSequenceClassification` (masked-mean or
  first-token pooling, then a prediction head and a linear classifier).
* Importing `lr_attnres_bert` registers the classes with `AutoConfig` / `AutoModel` /
  `AutoModelForMaskedLM` / `AutoModelForSequenceClassification`, and
  `save_pretrained` writes an `auto_map` and copies the modeling code next to the
  weights, so a saved directory or Hub repo also loads without this package via
  `AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)`.
* Pass `attention_mask` for padded batches. When nothing is padded the model drops
  the mask (in eager mode), so SDPA can use its flash kernel.
* `examples/quickstart.py` runs fill-mask, sentence embeddings and the
  classification head (`--model` for a checkpoint; random weights otherwise).

## Speed and memory

Measured with `benchmarks/benchmark_inference.py` on one **RTX 5080 (16 GB)**:
bf16 weights, `torch.inference_mode`, the **encoder forward pass** (`last_hidden_state`;
no vocabulary projection), full-length inputs (no padding), median of 20 forwards
after 5 warm-up ones. Three models:

* **LR-AttnRes-BERT** — the default configuration (random weights: speed and memory
  do not depend on their values; non-zero queries, so the routed path is the one
  timed);
* **same, LR-AttnRes off** — the identical configuration with `attnres_rank=0`, the
  plain residual encoder: the difference between the two is exactly what LR-AttnRes
  costs;
* **ModernBERT-base** — `answerdotai/ModernBERT-base` through `transformers` with SDPA
  attention, the reference at the same depth and width. flash-attn is not installed
  here, so its sliding-window layers run as masked SDPA; with flash-attn (unpadded,
  windowed kernels) ModernBERT is faster than shown, especially at long lengths.

_NVIDIA GeForce RTX 5080, torch 2.12.1+cu132 (CUDA 13.2), transformers 5.12.1, bf16, median of 20 forwards after 5 warm-up._

**Latency (ms) and throughput (k tokens/s), eager**

| batch x length | LR-AttnRes-BERT | same, LR-AttnRes off | ModernBERT-base | LR-AttnRes overhead |
|---|---:|---:|---:|---:|
| 1 x 512 | 17.03 (30k) | 8.86 (58k) | 9.83 (52k) | +92.2% |
| 64 x 128 | 41.65 (197k) | 29.78 (275k) | 44.77 (183k) | +39.9% |
| 32 x 512 | 91.81 (178k) | 70.33 (233k) | 96.06 (171k) | +30.5% |
| 8 x 2048 | 123.20 (133k) | 100.85 (162k) | 129.46 (127k) | +22.2% |
| 2 x 8192 | 251.53 (65k) | 229.77 (71k) | 273.95 (60k) | +9.5% |

**Latency (ms) and throughput (k tokens/s), compile**

| batch x length | LR-AttnRes-BERT | same, LR-AttnRes off | ModernBERT-base | LR-AttnRes overhead |
|---|---:|---:|---:|---:|
| 1 x 512 | 4.64 (110k) | 4.18 (122k) | 4.67 (110k) | +10.9% |
| 64 x 128 | 31.16 (263k) | 30.95 (265k) | 32.68 (251k) | +0.7% |
| 32 x 512 | 72.11 (227k) | 60.89 (269k) | 76.53 (214k) | +18.4% |
| 8 x 2048 | 103.89 (158k) | 91.65 (179k) | 166.22 (99k) | +13.4% |
| 2 x 8192 | 232.49 (70k) | 220.56 (74k) | 562.36 (29k) | +5.4% |

**Peak memory (GiB, weights included), eager**

| batch x length | LR-AttnRes-BERT | same, LR-AttnRes off | ModernBERT-base |
|---|---:|---:|---:|
| weights only | 0.287 | 0.287 | 0.287 |
| 1 x 512 | 0.340 | 0.294 | 0.299 |
| 64 x 128 | 0.664 | 0.389 | 0.463 |
| 32 x 512 | 1.010 | 0.491 | 0.639 |
| 8 x 2048 | 1.011 | 0.491 | 0.644 |
| 2 x 8192 | 1.013 | 0.493 | 1.311 |

**Peak memory (GiB, weights included), compile**

| batch x length | LR-AttnRes-BERT | same, LR-AttnRes off | ModernBERT-base |
|---|---:|---:|---:|
| weights only | 0.287 | 0.287 | 0.287 |
| 1 x 512 | 0.342 | 0.294 | 0.310 |
| 64 x 128 | 1.143 | 0.401 | 0.442 |
| 32 x 512 | 2.000 | 0.502 | 0.889 |
| 8 x 2048 | 2.000 | 0.502 | 2.061 |
| 2 x 8192 | 2.000 | 0.502 | 6.748 |

How to read it:

* **Use `torch.compile`.** Every routed read adds a handful of small kernels (44 reads
  in all), so in eager mode LR-AttnRes costs +10% at 2 × 8192 but roughly doubles the
  latency of a single 512-token sequence, where kernel launches dominate. Compiled,
  the overhead is +5% at 2 × 8192, +13–18% at 512–2048 tokens per sequence, and within noise at 64 × 128.
* **Memory grows with the token count, not the length.** A forward keeps up to nine
  residual sources (the embedding and the closed groups) plus the mixed reads of the
  current segment alive for every token: the peak is the same for 32 × 512, 8 × 2048
  and 2 × 8192 (16 384 tokens each), and the attention itself stays O(T) (SDPA's
  flash path). ModernBERT's masked sliding-window SDPA grows with the length instead.
* The compiled 1 × 512 cells are overhead-bound and vary by ~1 ms run to run (the
  plain model's cell is the mean of two 100-iteration runs: 3.62 and 4.75 ms).

Reproduce:

```bash
python benchmarks/benchmark_inference.py            # -> benchmarks/results/cuda.json
python benchmarks/make_tables.py benchmarks/results/cuda.json
```

`--models`, `--modes`, `--shapes`, `--dtype` and `--device cpu` select a subset.

## Tests

```bash
pip install -e ".[test]"
python -m pytest -q tests
```

CPU tests on a tiny configuration: the ModernBERT-base-scale defaults (≈150 M
parameters, tied decoder, group boundaries), the read sites, a routed read against a
direct computation, the zero-query reads matching the plain encoder, padding never
reaching real tokens, gradients reaching every parameter, bf16 inference, the
`tiny` preset, the stream staying fp32 under autocast, and the `save_pretrained` →
`from_pretrained` / `AutoModel*` round trip being exact.

## Layout

```
src/lr_attnres_bert/
  configuration_lr_attnres_bert.py   LRAttnResBertConfig
  modeling_lr_attnres_bert.py        the encoder, LR-AttnRes stream, heads
examples/quickstart.py               fill-mask, embeddings, classification head
benchmarks/benchmark_inference.py    speed / memory measurement -> results/*.json
benchmarks/make_tables.py            results JSON -> the tables in this README
benchmarks/results/                  raw measurements
tests/test_modeling.py               CPU tests
```

## References

* J. Su. *Low-Rank Attention Residuals.* arXiv:2607.09694, 2026 — sliced low-rank
  routing keys and Block AttnRes.
* B. Warner et al. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder
  for Fast, Memory Efficient, and Long Context Finetuning and Inference* (ModernBERT).
  arXiv:2412.13663, 2024.
* J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints.* arXiv:2305.13245, 2023.
* J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
  arXiv:2104.09864, 2021.
* D. So et al. *Primer: Searching for Efficient Transformers for Language Modeling*
  (squared ReLU). arXiv:2109.08668, 2021.

## License

MIT — see [LICENSE](LICENSE).
