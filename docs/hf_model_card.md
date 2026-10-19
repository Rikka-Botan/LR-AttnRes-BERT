---
license: mit
library_name: transformers
pipeline_tag: fill-mask
language:
  - en
tags:
  - custom_code
  - bert
  - masked-lm
  - attention-residuals
  - lr-attnres
datasets:
  - openbmb/Ultra-FineWeb-L1
  - HuggingFaceTB/dclm-edu
  - openbmb/UltraData-Code
---

# LR-AttnRes-BERT-tiny

A 29.8 M-parameter BERT-style masked-language model: **grouped-query attention +
RoPE + a squared-ReLU MLP**, with **Block Sliced Low-Rank Attention Residuals
(LR-AttnRes)** in place of the residual sum. This is the `tiny` preset of
[LR-AttnRes-BERT](https://github.com/Rikka-Botan/LR-AttnRes-BERT), pretrained on
1 B tokens at 1,024 tokens of context.

It is a **research checkpoint** from a controlled ablation against the identical
model without LR-AttnRes. Use it for masked-token prediction, as a small backbone to
fine-tune, or to study LR-AttnRes. It has not been fine-tuned or evaluated on
downstream tasks.

| | |
|:--|:--|
| architecture | 8 layers × 384, GQA 6 query / 2 KV heads (head dim 64), RoPE θ = 160,000, ReLU² MLP 1,152, RMSNorm pre-norm, no biases |
| residual | Block Sliced LR-AttnRes: rank 64, 8 groups, every read routed (16 pseudo-queries, 1,024 parameters) |
| parameters | 29,770,688 (MLM, tied decoder) / 29,572,480 (encoder) |
| tokenizer | ModernBERT's (vocabulary 50,368; `[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283, `[MASK]` 50284) |
| context | trained at 1,024 tokens; RoPE accepts up to 8,192, but quality falls sharply past 2,048 (see *Evaluation*) |
| weights | fp32 safetensors (119 MB) |
| code | `trust_remote_code=True`, or `pip install` the [GitHub repository](https://github.com/Rikka-Botan/LR-AttnRes-BERT) |

## Usage

Requires `transformers >= 5.0` and `torch`. Every output below was produced by
running the snippet as written, on CPU in fp32. On an RTX 5080, bf16 and
`torch.compile` return the same top-5 tokens.

**Fill-mask pipeline**

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="RikkaBotan/LR-AttnRes-BERT-tiny", trust_remote_code=True)
for r in fill("Paris is the [MASK] of France.", top_k=5):
    print(f"{r['token_str'].strip():>8}  {r['score']:.3f}")
```

```text
 capital  0.050
    name  0.041
    part  0.025
    city  0.023
   heart  0.023
```

**Masked-LM logits, batched**

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "RikkaBotan/LR-AttnRes-BERT-tiny"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True).eval()

texts = ["The capital of France is [MASK].",
         "I like to drink [MASK] in the morning.",
         "def add(a, b):\n    return a [MASK] b"]
batch = tok(texts, padding=True, return_tensors="pt")
with torch.inference_mode():
    logits = model(**batch).logits                       # (3, T, 50368)

for i in range(len(texts)):
    pos = (batch["input_ids"][i] == tok.mask_token_id).nonzero()[0, 0]
    top = logits[i, pos].softmax(-1).topk(5)
    print([f"{tok.decode([t]).strip()} {p:.3f}" for p, t in zip(top.values.tolist(), top.indices.tolist())])
```

```text
['France 0.034', 'here 0.024', 'Italy 0.019', 'Germany 0.018', 'today 0.015']
['coffee 0.127', 'more 0.085', 'it 0.055', 'tea 0.054', 'water 0.046']
['+ 0.302', ', 0.253', 'as 0.094', '- 0.090', '* 0.066']
```

**Sentence embeddings (masked mean pooling)**

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "RikkaBotan/LR-AttnRes-BERT-tiny"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()  # encoder only

sentences = ["A man is playing a guitar.",
             "Someone is playing an instrument.",
             "The stock market fell sharply today."]
batch = tok(sentences, padding=True, return_tensors="pt")
with torch.inference_mode():
    h = model(**batch).last_hidden_state                 # (3, T, 384)
m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
emb = F.normalize((h * m).sum(1) / m.sum(1), dim=-1)     # masked mean pooling
print(emb.shape)
print((emb @ emb.T).round(decimals=3))
```

```text
torch.Size([3, 384])
tensor([[1.0000, 0.9800, 0.9420],
        [0.9800, 1.0000, 0.9430],
        [0.9420, 0.9430, 1.0000]])
```

`AutoModel` reports the three unused MLM-head tensors as `UNEXPECTED`; that is
expected. These are raw masked-LM features: fine-tune before using them for
retrieval or similarity. `AutoModelForSequenceClassification` puts a new
classification head on the encoder for fine-tuning.

For GPU inference, load with `dtype=torch.bfloat16` and wrap the model in
`torch.compile`. The GitHub repository has runnable scripts in `examples/` and
speed and memory measurements.

## Training

| | |
|:--|:--|
| data | English web + code: Ultra-FineWeb-L1 (six CC-MAIN-2025 snapshots) 55 %, DCLM-Edu 35 %, UltraData-Code (L3, 11 programming languages) 10 %, shares exact in tokens |
| examples | one document per example, ≤ 1,024 tokens, batches padded dynamically |
| tokens | 1 B content tokens, a single phase (no long-context extension) |
| objective | masked LM, constant 30 % masking (80 / 10 / 10) |
| batch | 512 sequences (global) |
| optimizer | AdamW, β = (0.9, 0.95), ε = 10⁻⁸, weight decay 0.1 (0 on the LR-AttnRes queries), gradient clip 1.0 |
| learning rate | peak 6 × 10⁻⁴; linear warmup over 6 % of the steps (at least 500); cosine decay to 10 % |
| precision / hardware | bf16 autocast, `torch.compile` / 8 × RTX 5090 |
| seed | 1234 |

## Evaluation

**Setup.** This model was compared with an otherwise identical model without
LR-AttnRes (`attnres_rank = 0`; 1,024 fewer parameters), trained with the same
recipe, data order and seed.

* **Data.** Held-out text that neither model saw: a disjoint 0.1 % slice of the same
  mix.
* **Windows.** For each length from 8 to 65,536 tokens there are 20 measurements of
  65,536 tokens each, all in full windows.
* **Masking.** 15 % of the non-special tokens are replaced by `[MASK]` from a fixed
  seed, so both models see identical masks.
* **Metrics.** Pseudo-perplexity (exp of the mean masked-token cross-entropy; Salazar
  et al., 2020) and top-1 MLM accuracy. Each value is the mean ± 95 % Student-t CI.

**Table 1.** Masked-LM quality versus sequence length. Both models were trained at
1,024 tokens; † marks extrapolation. Bold marks the better model where the two
intervals do not overlap.

| Metric | Model | 8 | 32 | 128 | 512 | 1,024 | 2,048† | 4,096† | 8,192† |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Pseudo-perplexity ↓ | **this model** | **262.50 ± 4.02** | **42.66 ± 0.92** | 19.88 ± 0.53 | 15.34 ± 0.42 | 14.67 ± 0.46 | 16.27 ± 0.63 | 46.32 ± 3.40 | 1136.03 ± 77.73 |
| | plain residual | 473.08 ± 8.70 | 50.00 ± 1.15 | 20.97 ± 0.58 | 15.97 ± 0.42 | 15.13 ± 0.49 | 16.73 ± 0.65 | **36.16 ± 2.34** | **633.38 ± 47.92** |
| MLM accuracy (%) ↑ | **this model** | **19.91 ± 0.20** | **38.17 ± 0.25** | 47.66 ± 0.34 | 51.20 ± 0.37 | 51.91 ± 0.42 | 50.68 ± 0.44 | 40.03 ± 0.84 | 9.55 ± 0.34 |
| | plain residual | 15.40 ± 0.23 | 36.78 ± 0.24 | 47.07 ± 0.39 | 50.60 ± 0.31 | 51.44 ± 0.42 | 50.25 ± 0.47 | **42.22 ± 0.81** | **13.38 ± 0.55** |

![Pseudo-perplexity and MLM accuracy versus sequence length](https://huggingface.co/RikkaBotan/LR-AttnRes-BERT-tiny/resolve/main/figures/fig1_quality_vs_length.png)

**Figure 1.** Masked-LM quality versus sequence length. (a) Pseudo-perplexity, log
scale. (b) MLM accuracy. Error bars are 95 % CIs. The shaded region is beyond the
training length.

![LR-AttnRes minus the plain residual versus sequence length](https://huggingface.co/RikkaBotan/LR-AttnRes-BERT-tiny/resolve/main/figures/fig2_difference_vs_length.png)

**Figure 2.** This model minus the plain residual. (a) Relative pseudo-perplexity
change (%). (b) MLM accuracy change (points). The CIs are conservative: they treat
the two models as independent, although the masks are shared.

* **Within the training length, LR-AttnRes is better, most of all on short inputs.**
  Pseudo-perplexity is −44.5 % at 8 tokens, −14.7 % at 32 and −5.2 % at 128, and MLM
  accuracy is +4.5, +1.4 and +0.6 points. From 256 to 2,048 tokens the gap narrows
  to about −3 % and +0.5 points, and at 1,024 and 2,048 it is within the CI.
* **Beyond twice the training length, it extrapolates worse.** Pseudo-perplexity is
  +28 % at 4,096 and +79 % at 8,192, and MLM accuracy is −2.2 and −3.8 points. By
  32,768 tokens both models are at 2–4 % accuracy.

## Limitations

* **Small and briefly trained.** 29.8 M parameters and 1 B tokens: predictions often
  get the type of a word right but the fact wrong, as in "The capital of France is
  `[MASK]`" above.
* **Inputs up to about 2,048 tokens.** The model saw at most 1,024 tokens per example.
* **One training run per arm.** The CIs cover evaluation sampling, not training-seed
  variance, so differences of about half a point need more seeds before they can be
  relied on.
* **No downstream evaluation.** Nothing here measures GLUE or retrieval quality.
* **Web-crawled training data.** The model can reproduce the biases and errors of its
  web and code corpora.

## Citation

```bibtex
@misc{lr-attnres-bert,
  title        = {{LR-AttnRes-BERT}: A {BERT}-style encoder with Block Sliced Low-Rank Attention Residuals},
  author       = {Rikka Botan},
  year         = {2026},
  howpublished = {\url{https://github.com/Rikka-Botan/LR-AttnRes-BERT}}
}
```

LR-AttnRes follows J. Su, *Low-Rank Attention Residuals*, arXiv:2607.09694, 2026.
Pseudo-perplexity follows J. Salazar et al., *Masked Language Model Scoring*,
ACL 2020.

## License

MIT
