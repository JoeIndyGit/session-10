from pathlib import Path
import json
import math
import os
import nbformat

root = Path(__file__).parent
metrics_path = root / "artifacts" / "metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

# Schema guard: the checked metrics must match the current experiment layout.
assert metrics.get("schema_version") == 2, "Stale metrics schema: execute the notebook again."

# 1) Numerical derivative must agree with autograd.
grad = metrics["gradient_check"]
assert grad["relative_error"] < 1e-6
assert math.isclose(
    abs(grad["autograd"] - grad["finite_difference"]),
    grad["absolute_error"],
    rel_tol=1e-9,
    abs_tol=1e-15,
)

# 2) Accumulation must truly involve unequal token counts and show a practical consequence.
acc = metrics["gradient_accumulation"]
assert acc["short_tokens"] != acc["long_tokens"]
assert acc["wrong_minus_correct_gap"] > 0

# 3) Strong objective proof: same distribution, unequal lengths, exact reference identity.
proof = acc["reference_gradient_proof"]
assert proof["same_distribution"] is True
assert proof["proof_short_tokens"] != proof["proof_long_tokens"]
assert proof["correct_cosine_similarity"] > 0.999999
assert proof["correct_max_abs_error"] < 1e-6
assert proof["correct_relative_l2_error"] < 1e-6
assert proof["wrong_relative_l2_error"] > 1e-3
assert proof["wrong_cosine_similarity"] < 0.999999
assert proof["wrong_relative_l2_error"] > proof["correct_relative_l2_error"] + 1e-3

# 4) Grad-norm trace must have a finite leading-signal example and every step logged.
lead = metrics["grad_norm_leading_signal"]
assert lead["trace_steps"] >= 50
assert 1 <= lead["step"] < lead["trace_steps"]
assert all(math.isfinite(float(lead[k])) for k in (
    "relative_grad_norm_move",
    "same_step_probe_loss_move",
    "next_3_step_probe_loss_move",
))
assert lead["relative_grad_norm_move"] > lead["same_step_probe_loss_move"]

# 5) MFU: numerator, denominator, provenance, 40% gap and sweep must all be explicit.
mfu = metrics["mfu"]
assert mfu["reported_mfu"] is not None and 0 < mfu["reported_mfu"] < 1
assert mfu["reported_peak_tflops"] is not None and mfu["reported_peak_tflops"] > 0
assert mfu["reported_peak_source"]
assert mfu["mfu_kind"]
assert math.isclose(
    mfu["reported_mfu"],
    mfu["estimated_model_tflops"] / mfu["reported_peak_tflops"],
    rel_tol=1e-10,
)
assert math.isclose(
    mfu["gap_to_40_percentage_points"],
    (0.40 - mfu["reported_mfu"]) * 100,
    rel_tol=1e-10,
)
sweep = mfu["batch_size_sweep"]
assert len(sweep) >= 5
assert all(x["batch_size"] > 0 and x["tflops"] > 0 for x in sweep)
best = max(sweep, key=lambda x: x["tflops"])
assert best["batch_size"] == mfu["best_sweep_batch_size"]
assert math.isclose(best["tflops"], mfu["best_sweep_tflops"], rel_tol=1e-12)
assert max(x["tflops"] for x in sweep) > min(x["tflops"] for x in sweep)

# 6) Bit patterns, stored values, quantization-error ordering and training choice.
f = metrics["float_0_1"]
assert f["fp32_bits"] == "00111101110011001100110011001101"
assert f["bf16_bits"] == "0011110111001101"
assert f["fp8_e4m3fn_bits"] == "00011101"
assert f["fp32_absolute_error"] < f["bf16_absolute_error"] < f["fp8_e4m3fn_absolute_error"]
assert f["recommended_training_format"].lower() == "bf16"

# 7) Core repository files must always be present.
core_required = [
    root / "training_loop_truth.ipynb",
    root / "training_loop_experiments.py",
    root / "README.md",
    root / "artifacts" / "metrics.json",
    root / ".github" / "workflows" / "verify.yml",
]
for path in core_required:
    assert path.exists(), f"Missing required file: {path.relative_to(root)}"
    assert path.stat().st_size > 0, f"Empty required file: {path.relative_to(root)}"

# Generated plots are required after a clean CI execution, but do not have to be committed as binaries.
generated = [
    root / "artifacts" / "gradient_accumulation_wrong_vs_correct.png",
    root / "artifacts" / "gradient_reference_proof.png",
    root / "artifacts" / "grad_norm_leading_signal.png",
    root / "artifacts" / "mfu_batch_sweep.png",
    root / "artifacts" / "experiment_summary.png",
]
if os.environ.get("CI"):
    for path in generated:
        assert path.exists(), f"Notebook did not generate: {path.relative_to(root)}"
        assert path.stat().st_size > 0, f"Generated artifact is empty: {path.relative_to(root)}"

# 8) Notebook is the executable entry point; implementation remains readable in the companion script.
nb = nbformat.read(root / "training_loop_truth.ipynb", as_version=4)
notebook_source = "\n".join(cell.source for cell in nb.cells)
script_source = (root / "training_loop_experiments.py").read_text(encoding="utf-8")
all_source = notebook_source + "\n" + script_source
for needle in (
    "reference_gradient_proof",
    "same_distribution",
    "batch_size_sweep",
    "reported_peak_source",
    "fp32_absolute_error",
    "experiment_summary.png",
):
    assert needle in all_source, f"Experiment source is missing: {needle}"
code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
assert code_cells and all(cell.execution_count is not None for cell in code_cells), "Notebook entry point is not executed."
assert "training_loop_experiments.py" in notebook_source, "Notebook must execute the experiment implementation."

print("Assignment verification passed.")
print(f"Gradient finite-difference relative error: {grad['relative_error']:.3e}")
print(f"Correct accumulation relative L2 error:    {proof['correct_relative_l2_error']:.3e}")
print(f"Broken accumulation relative L2 error:     {proof['wrong_relative_l2_error']:.2%}")
print(f"Accumulation validation-loss gap:          {acc['wrong_minus_correct_gap']:.6f}")
print(f"Leading-signal step:                       {lead['step']}")
print(f"Reported MFU:                              {mfu['reported_mfu']*100:.2f}%")
print(f"Best sweep batch:                          {mfu['best_sweep_batch_size']}")
