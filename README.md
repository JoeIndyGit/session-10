# The Training Loop — Make the Model Tell the Truth

> **One tiny transformer, six falsifiable experiments, and an observable training loop.**

![Training-loop experiment summary](artifacts/experiment_summary.png)

This repository turns a small language-model training loop into an observable system. The executed notebook generates the measurements, plots, and `artifacts/metrics.json`; `verify_assignment.py` independently checks the numerical invariants. GitHub Actions reruns the notebook from a clean kernel before running those checks so the repository stays reproducible.

## Results at a glance

The main results from the checked-in run are:

1. **Finite-difference gradient check:** `backward()` agrees with an independently nudged scalar weight to relative error **9.66e-11**.
2. **Gradient-accumulation identity test:** correct token-weighted accumulation has **0.0e+00 relative L2 error** against the exact combined-token objective; broken average-of-averages has **43.57%** error and cosine **0.921247**.
3. **Learning-curve consequence:** the broken loop finishes **0.043614 loss worse** on fixed token-weighted evaluation.
4. **Earlier optimization signal:** at step **57**, global grad norm moves **14.38%** while fixed probe loss moves only **0.038%**.
5. **MFU/performance accounting:** baseline allocated-CPU roofline MFU estimate is **2.29%**, a **37.71 percentage-point** gap to 40%; the batch sweep reaches **5.59% at B=64**.
6. **0.1 really is different in each format:** fp32 `0x3DCCCCCD`, bf16 `0x3DCD`, fp8 E4M3 `0x1D`; quantization error grows from **1.490e-09 → 9.766e-05 → 1.562e-03**.

Run the consistency checks:

```bash
python verify_assignment.py
```

---

## Experiment overview

| Experiment | Method | Checked-in result |
|---|---|---|
| Print every tensor shape + explain dimensions | Explicit attention/MLP forward path, flattened CE tensors, parameter/gradient shapes | `B,T,C,H,D,F,V` defined and every printed tensor gets a one-line axis explanation |
| Verify one gradient by hand | Central finite difference on one scalar `output_head.weight` | autograd `0.500315215386` vs finite difference `0.500315215435`; rel. error **9.66e-11** |
| Break gradient accumulation | Unequal 96-token / 384-token micro-batches plus wrong/correct learning curves | final loss gap **0.043614** |
| Prove which accumulation is correct | Same-distribution unequal-length proof batches; exact scalar reference objective | correct rel-L2 **0.0e+00**; broken **43.57%** |
| Log grad norm every step | Global L2 grad norm + train loss + fixed probe loss over 90 steps | step **57**: grad **14.38%** vs loss **0.038%** |
| Compute MFU + explain gap to 40% | Explicit model-FLOP numerator, exposed CPU roofline denominator, empirical GEMM sanity check, batch sweep | baseline **2.29%**, best observed sweep **5.59%** |
| Write 0.1 in fp32/bf16/fp8 E4M3 | Binary normalization, exponent biasing, round-to-nearest reasoning, bits, hex, stored values, errors | fp32 `0x3DCCCCCD`; bf16 `0x3DCD`; E4M3 `0x1D` |

---

## 1. Shape truth: no anonymous axes

The model is a one-block character-level transformer with attention written explicitly.

- **B** — batch size
- **T** — sequence length / prediction positions
- **C** — model width
- **H** — attention heads
- **D** — head width, `C/H`
- **F** — feed-forward hidden width
- **V** — vocabulary size

Representative tensors:

| Tensor | Shape | Meaning |
|---|---:|---|
| `tokens` | `[B,T]` | B sequences, T token positions each |
| `token_embedding` | `[B,T,C]` | C-dimensional token representation |
| `q_heads`, `k_heads`, `v_heads` | `[B,H,T,D]` | H attention heads of width D |
| `attention_scores` | `[B,H,T,T]` | query-to-key scores per batch/head |
| `causal_mask` | `[T,T]` | future-token visibility mask |
| `context` | `[B,T,C]` | heads concatenated back to model width |
| `mlp_hidden` | `[B,T,F]` | feed-forward expansion |
| `logits` | `[B,T,V]` | vocabulary scores per position |
| `flat_logits` | `[B*T,V]` | CE prediction rows |
| `flat_targets` | `[B*T]` | one target class per row |
| `loss` | `[]` | scalar mean cross entropy |

After `backward()`, the notebook also prints every parameter shape beside its gradient shape.

---

## 2. One derivative, independently measured

For scalar loss `L` and weight `w`:

```text
dL/dw ≈ [L(w + ε) - L(w - ε)] / (2ε)
```

The check runs in float64 with `ε=1e-5` so numerical-difference noise does not dominate.

```text
chosen weight       output_head.weight[49, 26]
backward()          0.500315215386
finite difference   0.500315215435
absolute error      4.833e-11
relative error      9.661e-11
```

The notebook fails loudly if relative error exceeds `1e-6`.

---

## 3. Gradient accumulation: break it, plot it, then prove it

### The bug

```text
short micro-batch   8 × 12 = 96 tokens
long micro-batch    8 × 48 = 384 tokens
```

Broken average-of-averages gives the batches **50% / 50%** weight:

```python
loss = (short_loss.mean() + long_loss.mean()) / 2
```

Correct token weighting gives **20% / 80%**:

```python
loss = (short_loss.sum() + long_loss.sum()) / (96 + 384)
```

So each short-batch token is overweighted **4× relative to a long-batch token** under the broken rule.

### Exact reference-gradient identity test

The proof deliberately samples *both* unequal-length micro-batches from the same prose distribution, removing domain/style as an explanation. Three identical models receive identical examples:

```text
REFERENCE = backward((sum_short + sum_long) / total_tokens) once
CORRECT   = backward(sum_short / total_tokens), then backward(sum_long / total_tokens)
BROKEN    = backward(mean_short / 2), then backward(mean_long / 2)
```

| Comparison | Cosine similarity | Max abs. gradient error | Relative L2 error |
|---|---:|---:|---:|
| Correct vs reference | **1.000000000000** | **0.000e+00** | **0.000e+00** |
| Broken vs reference | **0.921246767044** | **5.237e-02** | **43.57%** |

![Gradient reference proof](artifacts/gradient_reference_proof.png)

The correct accumulated gradient is the reference gradient. The average-of-averages rule follows a different gradient field.

### Training consequence

![Wrong versus correct gradient accumulation](artifacts/gradient_accumulation_wrong_vs_correct.png)

```text
wrong final loss     3.024153
correct final loss   2.980539
wrong - correct      0.043614
```

The curve is the consequence; the gradient identity test is the proof.

---

## 4. Grad norm as an earlier sensor

Every step logs training loss, fixed-probe loss and global L2 gradient norm.

At automatically selected step **57**:

```text
relative grad-norm move              14.38%
same-step fixed probe-loss move      0.038%
probe-loss move within next 3 steps  0.743%
```

![Gradient norm leading signal](artifacts/grad_norm_leading_signal.png)

Loss answers “how wrong are the predictions?” Grad norm answers “how hard is optimization trying to move the parameters?” They are different sensors, which is exactly why the norm can move first.

---

## 5. MFU: expose both numerator and denominator

Dominant forward matmul FLOPs/token are estimated as:

```text
QKV projections              6C²
attention output projection  2C²
MLP up + down                4CF
vocabulary head              2CV
QKᵀ + attention×V             4TC
---------------------------------
forward ≈ 8C² + 4CF + 2CV + 4TC
training ≈ 3 × forward
```

```text
MFU = estimated model FLOPs/sec / peak FLOPs/sec
```

### Checked-in run

```text
mean step time              2.800 ms
tokens/second               182,857
model throughput            0.013482 TFLOP/s
reported peak denominator   0.588800 TFLOP/s
reported roofline MFU       2.29%
gap to 40%                  37.71 percentage points
empirical GEMM ceiling      0.229 TFLOP/s
empirical ceiling util.     5.89%
```

**Denominator provenance** saved in `metrics.json`:

> analytical allocated-CPU FP32 roofline: 4 PyTorch threads × 2.300 GHz × 16 fp32 lanes (AVX-512) × 2 FLOP/lane-instruction × 2 assumed vector issue units/core

This is explicitly labeled an **analytical allocated-CPU roofline estimate**, not a vendor-published accelerator peak. On a GPU, set `THEORETICAL_PEAK_TFLOPS` to the vendor peak for the exact GPU and precision; the same formula then becomes canonical accelerator MFU.

### Performance sweep

The benchmark uses median timing across multiple measurement rounds to reduce noisy-run bias.

| Batch | Tokens/s | Model TFLOP/s | Roofline MFU |
|---:|---:|---:|---:|
| 4 | 75,877 | 0.00559 | 0.95% |
| 8 | 133,653 | 0.00985 | 1.67% |
| 16 | 222,174 | 0.01638 | 2.78% |
| 32 | 311,002 | 0.02293 | 3.89% |
| 64 | 446,230 | 0.03290 | 5.59% |

![MFU batch sweep](artifacts/mfu_batch_sweep.png)

Best observed point in this run: **B=64**, **0.03290 TFLOP/s**, **5.59% MFU**. The sweep need not be perfectly monotonic on a shared CPU runtime; the important point is that larger useful matrix work materially improves utilization over the tiny-batch case.

### Why am I far from 40%?

- tiny GEMMs under-fill wide compute units,
- eager/Python dispatch is large relative to useful arithmetic,
- optimizer, softmax, layer norm, indexing and memory traffic cost wall time without being fully credited in the simplified FLOP numerator,
- no fused attention or graph compilation is used in the transparent baseline,
- the model and sequence sizes are pedagogical rather than saturation-oriented.

On suitable accelerator hardware I would test BF16 autocast, larger effective batch/model/sequence shapes, fused scaled-dot-product attention, `torch.compile`, efficient optimizer kernels, fewer synchronizations and input overlap—measuring after every change.

---

## 6. Decimal 0.1: show the rounding, not just the hex

```text
0.1₁₀ = 0.00011001100110011…₂
      = 1.1001100110011…₂ × 2⁻⁴
```

The `0011` tail repeats forever.

| Format | Stored bits | Hex | Stored value | Absolute error |
|---|---|---|---:|---:|
| fp32 | `0 01111011 10011001100110011001101` | `0x3DCCCCCD` | 0.10000000149011612 | 1.490e-09 |
| bf16 | `0 01111011 1001101` | `0x3DCD` | 0.100097656250 | 9.766e-05 |
| fp8 E4M3 | `0 0011 101` | `0x1D` | 0.1015625 | 1.562e-03 |

The notebook writes out each round-to-nearest decision explicitly. For example, E4M3 keeps `100` from `100110…`; the next bit is `1` with a non-zero tail, so it rounds to `101`.

### Which would I train in?

**BF16** on hardware with native support, while keeping sensitive accumulation/optimizer state in higher precision where appropriate. BF16 preserves the 8-bit exponent range of fp32 while reducing memory traffic; naive E4M3 is too coarse for an all-state training format without scaling and mixed-precision machinery.

For the finite-difference diagnostic, I deliberately use float64: the best precision for debugging a derivative is not necessarily the best precision for efficient training.

---

## Repository structure

```text
session-10/
├── README.md
├── training_loop_truth.ipynb
├── requirements.txt
├── verify_assignment.py
├── .gitignore
├── .github/
│   └── workflows/
│       └── verify.yml
└── artifacts/
    ├── experiment_summary.png
    ├── gradient_reference_proof.png
    ├── gradient_accumulation_wrong_vs_correct.png
    ├── grad_norm_leading_signal.png
    ├── mfu_batch_sweep.png
    └── metrics.json
```

## Reproduce from scratch

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter nbconvert \
  --to notebook \
  --execute training_loop_truth.ipynb \
  --output training_loop_truth.rerun.ipynb \
  --ExecutePreprocessor.timeout=300
python verify_assignment.py
```

### Clean-room CI check

`.github/workflows/verify.yml` copies the repository into an isolated directory, executes the notebook from a clean kernel, **replaces the notebook in that isolated copy with the newly executed result**, then runs the verifier. This catches stale generated files by ensuring the numerical checks run only after a fresh notebook execution.

## Reproducibility guardrails

- deterministic Python/PyTorch seed: `7`
- no network download required
- no dropout in the gradient-check path
- accumulation models start from identical parameters and consume identical examples
- reference proof uses the same data distribution and changes only valid-token count
- fixed token-weighted validation
- every grad norm logged across 90 steps
- median timing rounds for performance benchmarks
- headline evidence serialized by the notebook itself
- plots generated by the notebook itself
- notebook ends with assertions
- independent verifier checks schema + numerical invariants + artifact existence + full execution
- CI regenerates evidence before verification

---

## Final takeaway

A training loop can produce a beautiful loss curve while optimizing the wrong objective. The defense is not another abstraction layer; it is falsifiable observability:

**shape the tensors → nudge a weight → prove the accumulated gradient → watch the norm → account for the FLOPs → inspect the bits.**

The central idea is simple: make each important claim independently testable, then keep those tests close to the training loop.
