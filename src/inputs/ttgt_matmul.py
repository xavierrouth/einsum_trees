import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset18 as op
from onnxscript.onnx_types import FLOAT
from pathlib import Path  # added
import os  # added

@script()
def TTGTMatMul( 
    X: FLOAT["i", "h"],
    W: FLOAT["b", "i"],
) -> FLOAT["b", "h"]:
    return op.Einsum(X, W, equation="ih,bi->bh")

def _export_model():
    breakpoint()
    model = TTGTMatMul.to_model_proto()  # updated to new function
    out_path = Path(__file__).with_suffix(".model")
    onnx.save(model, out_path.as_posix())

_export_model()

