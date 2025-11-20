import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset18 as op
from onnxscript.onnx_types import FLOAT
from pathlib import Path  # added
import os  # added

#𝑝𝑞𝑟, 𝑠𝑝𝑟 → 𝑠𝑞𝑟
@script()
def PackedMatMul(
    X: FLOAT["p", "q", "r"],
    W: FLOAT["s", "p", "r"],
) -> FLOAT["s", "q", "r"]:
    return op.Einsum(X, W, equation="pqr,spr->sqr")

def _export_model():
    model = PackedMatMul.to_model_proto()  # updated to new function
    out_path = Path(__file__).with_suffix(".model")
    onnx.save(model, out_path.as_posix())

_export_model()

