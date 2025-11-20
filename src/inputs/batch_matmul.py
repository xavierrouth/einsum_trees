import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset18 as op
from onnxscript.onnx_types import FLOAT
from pathlib import Path  # added
import os  # added

def LoopBatchmatMul( 
    X: FLOAT[10, "q", "r"],
    W: FLOAT[10, "s", "q"],
) -> FLOAT[10, "s", "r"]:
    return op.Einsum(X, W, equation="pqr,psq->psr")
    
def _export_model():
    model = LoopBatchmatMul.to_model_proto()  # updated to new function
    out_path = Path(__file__).with_suffix(".model")
    onnx.save(model, out_path.as_posix())

_export_model()

