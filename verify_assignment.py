from pathlib import Path
import json
import math
import os
import nbformat

root = Path(__file__).parent
metrics = json.loads((root / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
assert metrics.get("schema_version") == 3, "Run the notebook again to regenerate the current metrics schema."

# 1) Numerical finite-difference derivative.
grad = metrics["gradient_check"]
assert grad["relative_error"] < 1e-6
assert math.isfinite(grad["autograd"]) and math.isfinite(grad["finite_difference"])

# 2) Unequal-length accumulation experiment + exact reference proof.
acc = metrics["gradient_accumulation"]
assert acc["short_tokens"] == 96 and acc["long_tokens"] == 384
assert acc["short_tokens"] != acc["long_tokens"]
assert acc["max_curve_gap"] > 1e-3
assert acc["correct_reference_rel_l2"] < 1e-6
assert acc["broken_reference_rel_l2"] > 1e-3

# 3) Grad norm is logged every step and a leading-signal step exists.
lead = metrics["grad_norm"]
assert lead["steps"] >= 50
assert 1 <= lead["leading_step"] < lead["steps"]
assert lead["grad_move"] > lead["same_step_loss_move"]
assert all(math.isfinite(float(lead[k])) for k in ("grad_move", "same_step_loss_move", "next_3_loss_move"))

# 4) MFU exposes both numerator and denominator and reports the gap to 40%.
mfu = metrics["mfu"]
assert mfu["estimated_model_tflops"] > 0
assert mfu["peak_tflops"] is not None and mfu["peak_tflops"] > 0
assert mfu["peak_source"]
assert mfu["reported_mfu"] is not None and 0 < mfu["reported_mfu"] < 1
assert math.isclose(mfu["reported_mfu"], mfu["estimated_model_tflops"] / mfu["peak_tflops"], rel_tol=1e-10)
assert math.isclose(mfu["gap_to_40_percentage_points"], (0.40 - mfu["reported_mfu"]) * 100, rel_tol=1e-10)

# 5) Hand-derived floating-point patterns and training choice.
f = metrics["float_0_1"]
assert f["fp32_bits"] == "00111101110011001100110011001101"
assert f["bf16_bits"] == "0011110111001101"
assert f["fp8_e4m3_bits"] == "00011101"
assert f["training_choice"].lower() == "bf16"

# 6) Notebook must itself contain every assignment step; no wrapper dependency.
nb = nbformat.read(root / "training_loop_truth.ipynb", as_version=4)
source = "\n".join(cell.source for cell in nb.cells)
for needle in (
    "class TinyGPT",
    "attention_scores",
    "finite_difference",
    "WRONG average-of-averages",
    "CORRECT token-weighted",
    "grad_norm",
    "MFU",
    "0.1_{10}",
    "3DCCCCCD",
    "3DCD",
    "0x1D",
):
    assert needle in source, f"Notebook is missing required source/evidence: {needle}"
assert "%run" not in source and "training_loop_experiments.py" not in source, "Notebook must be fully self-contained."
code_cells = [c for c in nb.cells if c.cell_type == "code"]
assert code_cells and all(c.execution_count is not None for c in code_cells), "Notebook should be checked in with executed cell order."

# 7) Clean CI execution must regenerate the required visual evidence.
if os.environ.get("CI"):
    for name in ("gradient_accumulation_wrong_vs_correct.png", "grad_norm_leading_signal.png"):
        path = root / "artifacts" / name
        assert path.exists() and path.stat().st_size > 0, f"Notebook did not generate {name}"

print("Assignment verification passed.")
print(f"Gradient relative error: {grad['relative_error']:.3e}")
print(f"Correct accumulation relative L2: {acc['correct_reference_rel_l2']:.3e}")
print(f"Broken accumulation relative L2: {acc['broken_reference_rel_l2']:.2%}")
print(f"Curve gap: {acc['max_curve_gap']:.6f}")
print(f"Leading grad-norm step: {lead['leading_step']}")
print(f"MFU: {mfu['reported_mfu']*100:.2f}%")
