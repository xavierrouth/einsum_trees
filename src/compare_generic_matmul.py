import argparse
import importlib
import importlib.util
from pathlib import Path
import time
import numpy as np

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
    import onnxruntime as ort
    so = ort.SessionOptions()
    return ort.InferenceSession(path.as_posix(), sess_options=so, providers=[provider])

def _list_nodes(model_path: Path):
    import onnx
    m = onnx.load(model_path.as_posix())
    ops = [n.op_type for n in m.graph.node]
    print(f"Model nodes ({len(ops)}): {ops}")

def main():
    ap = argparse.ArgumentParser(description="Generic ONNX vs NumPy matmul/einsum comparison script.")
    ap.add_argument("--input-module", default="inputs.packed_matmul",
                    help="Module name or path; must export model on import and define generate_tensors().")
    ap.add_argument("--nonopt-path", default=None)
    ap.add_argument("--opt-path", default=None)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--list-nodes", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # get sizes
    if args.input_module.endswith("batch_matmul.py"):
        def sizes():
            return {
                "p": 10,
                "q": 640,
                "r": 128,
                "s": 96,
            }
        # numpy test tensor generation
        S = sizes()
        p, q, r, s = S["p"], S["q"], S["r"], S["s"]
        np.random.seed(0)
        X = np.random.randn(p, q, r).astype(np.float32)
        W = np.random.randn(p, s, q).astype(np.float32)
        ref_fn = lambda X, W: np.einsum("pqr,psq->psr", X, W)
        ref = np.einsum("pqr,psq->psr", X, W)
    elif args.input_module.endswith("ttgt_matmul.py"):
        def sizes():
            return {
                "b": 640,
                "i": 128,
                "h": 96,
            }
        # numpy test tensor generation
        S = sizes()
        b, i, h = S["b"], S["i"], S["h"]
        np.random.seed(0)
        X = np.random.randn(i, h).astype(np.float32)
        W = np.random.randn(b, i).astype(np.float32)
        ref_fn = lambda X, W: np.einsum("ih,bi->bh", X, W)
        ref = np.einsum("ih,bi->bh", X, W)
    elif args.input_module.endswith("packed_matmul.py"):
        def sizes():
            return {
                "p": 10,
                "q": 640,
                "r": 128,
                "s": 96,
            }
        # numpy test tensor generation
        S = sizes()
        p, q, r, s = S["p"], S["q"], S["r"], S["s"]
        np.random.seed(0)
        X = np.random.randn(p, q, r).astype(np.float32)
        W = np.random.randn(s, p, r).astype(np.float32)
        ref_fn = lambda X, W: np.einsum("pqr,spr->sqr", X, W)
        ref = np.einsum("pqr,spr->sqr", X, W)
    else: 
        # error
        raise SystemExit("Unsupported input module; must be one of batch_matmul.py, ttgt_matmul.py, packed_matmul.py")

    model_path = Path(args.input_module).with_suffix(".model")
    nonopt_path = Path(args.nonopt_path) if args.nonopt_path else model_path
    opt_path = Path(args.opt_path) if args.opt_path else model_path.with_suffix(".optimized_model")

    if not nonopt_path.exists():
        raise SystemExit(f"Non-optimized model not found: {nonopt_path}")

    if args.list_nodes:
        _list_nodes(nonopt_path)

    sess_nonopt = _create_session(nonopt_path, args.provider)
    (y_nonopt,) = sess_nonopt.run(None, {"X": X, "W": W})
    diff_nonopt = float(np.max(np.abs(y_nonopt - ref)))

    if opt_path.exists():
        sess_opt = _create_session(opt_path, args.provider)
        (y_opt,) = sess_opt.run(None, {"X": X, "W": W})
        diff_opt = float(np.max(np.abs(y_opt - ref)))
        diff_between = float(np.max(np.abs(y_nonopt - y_opt)))
    else:
        y_opt = None

    print(f"Module: {args.input_module}")
    print(f"Shapes: X={X.shape}, W={W.shape}, Ref={ref.shape}")
    print(f"Non-optimized model: {nonopt_path}")
    if y_opt is not None:
        print(f"Optimized model: {opt_path}")
    print(f"Ref checksum: {float(ref.sum()):.6g}")
    print(f"Non-optimized checksum: {float(y_nonopt.sum()):.6g}")
    print(f"Max abs diff (NumPy vs non-optimized): {diff_nonopt:.6g}")
    if y_opt is not None:
        print(f"Optimized checksum: {float(y_opt.sum()):.6g}")
        print(f"Max abs diff (NumPy vs optimized): {diff_opt:.6g}")
        print(f"Max abs diff (non-optimized vs optimized): {diff_between:.6g}")
    else:
        print(f"Optimized model missing; expected at: {opt_path}")

    if not args.no_bench:
        print(f"Benchmarking provider={args.provider}, warmup={args.warmup}, runs={args.runs}")
        _bench("NumPy reference", lambda: ref_fn(X, W), args.warmup, args.runs)
        _bench("ORT non-optimized", lambda: sess_nonopt.run(None, {"X": X, "W": W}), args.warmup, args.runs)
        if y_opt is not None:
            _bench("ORT optimized", lambda: sess_opt.run(None, {"X": X, "W": W}), args.warmup, args.runs)

if __name__ == "__main__":
    main()
