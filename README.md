# The Training Loop — Make the Model Tell the Truth

A small transformer and a real training loop instrumented so the important assumptions are visible instead of hidden behind a trainer abstraction.

The notebook is the entry point. It executes `training_loop_experiments.py`, which prints tensor and gradient shapes, verifies a derivative numerically, deliberately breaks gradient accumulation, logs gradient norm, measures MFU, derives the representation of decimal `0.1`, and regenerates the experiment artifacts.

## Experiments

| Experiment | What is tested |
|---|---|
| Tensor shapes | Every intermediate tensor in the explicit attention/MLP forward pass, the flattened cross-entropy tensors, the scalar loss, and every parameter/gradient shape |
| Numerical gradient | One scalar weight is checked with a central finite difference and compared with `backward()` |
| Gradient accumulation | Unequal 96-token and 384-token micro-batches demonstrate the average-of-averages bug |
| Reference gradient | Correct sequential accumulation is compared with the exact combined-token objective on same-distribution data |
| Gradient norm | Global L2 grad norm is logged over 90 optimizer updates and compared with a fixed probe loss |
| MFU | Model FLOPs/s, peak-denominator provenance, the gap to 40%, empirical GEMM utilization, and a batch-size sweep |
| Floating point | `0.1` is derived in fp32, bf16, and fp8 E4M3 with bits, stored values, and quantization error |

## 1. Shape truth

The model is a one-block character-level transformer with attention written explicitly.

- **B** — batch size
- **T** — sequence length
- **C** — model width
- **H** — attention heads
- **D** — head width, `C/H`
- **F** — feed-forward hidden width
- **V** — vocabulary size

The run prints tensors such as `tokens [B,T]`, `q_heads [B,H,T,D]`, `attention_scores [B,H,T,T]`, `context [B,T,C]`, `mlp_hidden [B,T,F]`, `logits [B,T,V]`, `flat_logits [B*T,V]`, `flat_targets [B*T]`, and scalar `loss []`. After `backward()`, every parameter is printed beside its gradient shape.

## 2. Numerical gradient check

For scalar loss `L` and one selected weight `w`, the notebook checks:

```text
dL/dw ≈ [L(w + ε) - L(w - ε)] / (2ε)
```

The diagnostic runs in float64 with a small `ε`. The finite-difference derivative must agree with autograd to a relative error below `1e-6` or the experiment stops.

## 3. Break gradient accumulation on purpose

The two micro-batches contain different numbers of target tokens:

```text
short micro-batch   8 × 12 = 96 tokens
long micro-batch    8 × 48 = 384 tokens
```

The broken rule averages two already-averaged losses:

```python
loss = (short_loss.mean() + long_loss.mean()) / 2
```

That gives the two micro-batches equal 50/50 weight even though they contain 20/80 percent of the tokens.

The correct objective sums token losses and divides once by the total token count:

```python
loss = (short_loss.sum() + long_loss.sum()) / (96 + 384)
```

The notebook plots the wrong and correct training curves and then performs a stronger identity test: both accumulation rules are compared with one exact combined-token reference gradient. Correct accumulation must reproduce the reference; average-of-averages must not.

## 4. Gradient norm as another sensor

Every optimizer step records global gradient norm and a fixed probe loss. The notebook searches the trace for a point where gradient norm moves more sharply than the same-step probe loss, then reports what happens to loss over the following updates.

This is useful because loss and grad norm answer different questions: loss measures prediction error, while gradient norm measures how strongly the optimization objective is trying to move the parameters.

## 5. MFU and the gap to 40%

The model-FLOP numerator is written explicitly. The dominant forward matmul estimate per token is:

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

MFU is then:

```text
MFU = estimated model FLOPs/sec / peak FLOPs/sec
```

If an accelerator peak is supplied, the notebook uses that value. In the CPU fallback it reports a clearly labelled analytical roofline for the CPU resources allocated to the run. A measured GEMM ceiling is reported separately and is never presented as the theoretical peak.

A batch-size sweep shows how more useful matrix work can amortize the fixed overhead of this intentionally tiny model. Timing-dependent numbers will vary by machine; the exact run is written to `artifacts/metrics.json`.

## 6. Decimal 0.1

```text
0.1₁₀ = 0.00011001100110011…₂
      = 1.1001100110011…₂ × 2⁻⁴
```

The repeating binary expansion is rounded differently in each format:

| Format | Stored bits | Hex |
|---|---|---|
| fp32 | `0 01111011 10011001100110011001101` | `0x3DCCCCCD` |
| bf16 | `0 01111011 1001101` | `0x3DCD` |
| fp8 E4M3 | `0 0011 101` | `0x1D` |

Among these three, BF16 is the practical training choice on hardware with native support because it keeps the fp32-like exponent range while reducing memory and bandwidth cost. The finite-difference diagnostic deliberately uses float64 because debugging a derivative and efficiently training a model have different numerical requirements.

## Repository structure

```text
session-10/
├── README.md
├── training_loop_truth.ipynb       # notebook entry point
├── training_loop_experiments.py    # full readable implementation
├── requirements.txt
├── verify_assignment.py
├── .gitignore
├── .github/
│   └── workflows/
│       └── verify.yml
└── artifacts/
    ├── metrics.json                # representative checked-in run
    ├── experiment_summary.png      # generated by notebook
    ├── gradient_reference_proof.png
    ├── gradient_accumulation_wrong_vs_correct.png
    ├── grad_norm_leading_signal.png
    └── mfu_batch_sweep.png
```

The PNG files are generated outputs rather than source files; CI verifies that a clean notebook execution recreates them.

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

Or open `training_loop_truth.ipynb` and run its experiment cell directly.

## Reproducibility

- deterministic Python/PyTorch seed for the correctness experiments
- no external dataset or network download
- no dropout in the gradient-check path
- identical model initialization for wrong/correct accumulation comparisons
- same-distribution reference-gradient proof
- fixed token-weighted evaluation for the accumulation experiment
- grad norm recorded for every update in the trace
- multiple timing rounds for performance measurements
- machine-readable metrics generated from the run
- numerical assertions in the implementation
- independent consistency checks in `verify_assignment.py`
- GitHub Actions executes the notebook from a clean checkout before verification

## Takeaway

A training loop can produce a plausible loss curve while optimizing the wrong objective. The useful defense is falsifiable observability:

**shape the tensors → nudge a weight → prove the accumulated gradient → watch the norm → account for the FLOPs → inspect the bits.**
