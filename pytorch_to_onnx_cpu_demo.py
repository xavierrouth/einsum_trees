import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit(
        "onnxruntime is required to run this script on CPU.\n"
        "Install with: pip install onnxruntime\n"
        f"Details: {e}"
    )

class SimpleAdd(nn.Module):
    # y is scaled and added to x to form a simple op graph
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + 2.0 * y

def export_to_onnx(model: nn.Module, x: torch.Tensor, y: torch.Tensor, onnx_path: Path, opset: int = 13) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, y),
            onnx_path.as_posix(),
            input_names=["x", "y"],
            output_names=["out"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}, "out": {0: "batch"}},
            do_constant_folding=True,
            opset_version=opset,
        )

def run_onnx_cpu(onnx_path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"x": x, "y": y})
    return outputs[0]

def main():
    print(f"ONNX Runtime providers: {ort.get_available_providers()}")
    # breakpoint()
    parser = argparse.ArgumentParser(description="PyTorch -> ONNX -> onnxruntime (CPU) demo")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--features", type=int, default=3, help="Feature dimension")
    parser.add_argument("--onnx-path", type=str, default=None, help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    args = parser.parse_args()

    torch.manual_seed(0)
    batch, features = args.batch, args.features

    model = SimpleAdd().eval()
    x = torch.randn(batch, features, dtype=torch.float32)
    y = torch.randn(batch, features, dtype=torch.float32)

    onnx_path = Path(args.onnx_path) if args.onnx_path else Path(__file__).resolve().parent / "outputs" / "simple_add.onnx"
    export_to_onnx(model, x, y, onnx_path, opset=args.opset)

    with torch.no_grad():
        pt_out = model(x, y).cpu().numpy()
    ort_out = run_onnx_cpu(onnx_path, x.cpu().numpy(), y.cpu().numpy())

    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"Exported: {onnx_path}")
    print(f"Input shape: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"Output shape: {tuple(ort_out.shape)}")
    print(f"Max abs diff vs PyTorch: {max_abs_diff:.6g}")

if __name__ == "__main__":
    main()
