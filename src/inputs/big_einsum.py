import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset18 as op
from onnxscript.onnx_types import FLOAT
from pathlib import Path
import numpy as np
import argparse  # added
import time  # added

@script()
def BigEinsum(
    PQR: FLOAT["p", "q", "r"],
    PSQ: FLOAT["p", "s", "q"],
    SQ: FLOAT["s", "q"],
    ABC: FLOAT["a", "b", "c"],
    BCP: FLOAT["b", "c", "p"],
    # ABCPQR: FLOAT["a", "b", "c", "p", "q", "r"],
) -> FLOAT["p", "s", "r"]:
    return op.Einsum(PQR, PSQ, SQ, ABC, BCP, equation="pqr,psq,sq,abc,bcp->psr")

# Reasonable small sizes for a quick demo
def sizes():
    return {
        "a": 20,
        "b": 30,
        "c": 10,
        "p": 200,
        "q": 60,
        "r": 3,
        "s": 2,
    }

# numpy test tensor generation
def generate_tensors():
    S = sizes()
    a, b, c = S["a"], S["b"], S["c"]
    p, q, r, s = S["p"], S["q"], S["r"], S["s"]
    rng = np.random.default_rng(0)
    PQR = rng.standard_normal((p, q, r), dtype=np.float32)
    PSQ = rng.standard_normal((p, s, q), dtype=np.float32)
    SQ = rng.standard_normal((s, q), dtype=np.float32)
    ABC = rng.standard_normal((a, b, c), dtype=np.float32)
    BCP = rng.standard_normal((b, c, p), dtype=np.float32)
    # ABCPQR = rng.standard_normal((a, b, c, p, q, r), dtype=np.float32)
    return PQR, PSQ, SQ, ABC, BCP

def _export_model():
    model = BigEinsum.to_model_proto()
    out_path = Path(__file__).with_suffix(".model")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path.as_posix())
    return out_path

def _bench(label: str, fn, warmup: int, runs: int):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    dt = time.perf_counter() - t0
    avg_ms = (dt / max(runs, 1)) * 1000.0
    it_s = runs / dt if dt > 0 else float("inf")
    print(f"{label}: avg {avg_ms:.3f} ms, {it_s:.1f} it/s")

def _create_session(path: Path, provider: str):
    so = ort.SessionOptions()
    return ort.InferenceSession(path.as_posix(), sess_options=so, providers=[provider])

# numpy reference implementation
def big_einsum_np(PQR, PSQ, SQ, ABC, BCP):
    return np.einsum("pqr,psq,sq,abc,bcp->psr", PQR, PSQ, SQ, ABC, BCP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BigEinsum NumPy vs ONNX.")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    model_path = _export_model()
    opt_path = model_path.with_name(model_path.stem + "_optimized.model")
    opt2_path = model_path.with_name(model_path.stem + "_optimized2.model")  # added

    PQR, PSQ, SQ, ABC, BCP= generate_tensors()
    ref = big_einsum_np(PQR, PSQ, SQ, ABC, BCP)

    feeds = {
        "PQR": PQR,
        "PSQ": PSQ,
        "SQ": SQ,
        "ABC": ABC,
        "BCP": BCP
    }

    sess_nonopt = _create_session(model_path, args.provider)
    (ort_out_nonopt,) = sess_nonopt.run(None, feeds)
    diff_nonopt = float(np.max(np.abs(ref - ort_out_nonopt)))

    if opt_path.exists():
        sess_opt = _create_session(opt_path, args.provider)
        (ort_out_opt,) = sess_opt.run(None, feeds)
        diff_opt = float(np.max(np.abs(ref - ort_out_opt)))
        diff_between = float(np.max(np.abs(ort_out_nonopt - ort_out_opt)))
    else:
        sess_opt = None

    if opt2_path.exists():
        sess_opt2 = _create_session(opt2_path, args.provider)
        (ort_out_opt2,) = sess_opt2.run(None, feeds)
        diff_opt2 = float(np.max(np.abs(ref - ort_out_opt2)))
        diff_between2 = float(np.max(np.abs(ort_out_nonopt - ort_out_opt2)))
    else:
        sess_opt2 = None

    print(f"Saved model: {model_path}")
    if sess_opt:
        print(f"Optimized model: {opt_path}")
    if sess_opt2:
        print(f"Second optimized model: {opt2_path}")
    print(f"Output shape: {ort_out_nonopt.shape}")
    print(f"Ref checksum: {float(ref.sum()):.6g}")
    print(f"Non-optimized checksum: {float(ort_out_nonopt.sum()):.6g}")
    print(f"Max abs diff (NumPy vs non-optimized): {diff_nonopt:.6g}")
    if sess_opt:
        print(f"Optimized checksum: {float(ort_out_opt.sum()):.6g}")
        print(f"Max abs diff (NumPy vs optimized): {diff_opt:.6g}")
        print(f"Max abs diff (non-optimized vs optimized): {diff_between:.6g}")
    else:
        print(f"Optimized model missing; expected at: {opt_path}")
    if sess_opt2:
        print(f"Optimized2 checksum: {float(ort_out_opt2.sum()):.6g}")
        print(f"Max abs diff (NumPy vs optimized2): {diff_opt2:.6g}")
        print(f"Max abs diff (non-optimized vs optimized2): {diff_between2:.6g}")
        if sess_opt:
            print(f"Max abs diff (optimized vs optimized2): {float(np.max(np.abs(ort_out_opt - ort_out_opt2))):.6g}")
    else:
        print(f"Second optimized model missing; expected at: {opt2_path}")

    print(f"Benchmarking provider={args.provider}, warmup={args.warmup}, runs={args.runs}")
    _bench("NumPy reference", lambda: big_einsum_np(PQR, PSQ, SQ, ABC, BCP), args.warmup, args.runs)
    _bench("ONNX non-optimized", lambda: sess_nonopt.run(None, feeds), args.warmup, args.runs)
    if sess_opt:
        _bench("ONNX optimized", lambda: sess_opt.run(None, feeds), args.warmup, args.runs)
    if sess_opt2:
        _bench("ONNX optimized2", lambda: sess_opt2.run(None, feeds), args.warmup, args.runs)