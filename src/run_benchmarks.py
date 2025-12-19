#!/usr/bin/env python3
"""
Benchmark script comparing three approaches for einsum operations:
1. Unoptimized einsum expressions (direct ONNX Einsum op)
2. Binary contraction tree without data layout optimization
3. Binary contraction tree with data layout optimization

Results are dumped to a JSON file for later analysis.
"""

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

from main import lower_einsums


class BenchmarkEntry(NamedTuple):
    """Parsed benchmark entry from single_node.txt"""
    id: int
    contraction: str  # e.g., "abdfe,cf->abcde"
    dimension_sizes: list[int]
    arithmetic_intensity: float


def parse_benchmark_file(filepath: str) -> list[BenchmarkEntry]:
    """Parse benchmark files (single_node.txt or multi_node.txt).
    
    Format: ID,input1,input2,...,inputN->output,dim1,dim2,...,dimM,arithmetic_intensity
    
    The contraction string spans from parts[1] until we find the part containing '->'.
    After that, all remaining numeric values are dimension sizes except the last (intensity).
    """
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) < 4:
                continue
            
            try:
                benchmark_id = int(parts[0])
                
                # Find the part containing '->' which marks end of contraction
                arrow_idx = -1
                for i, part in enumerate(parts[1:], start=1):
                    if '->' in part:
                        arrow_idx = i
                        break
                
                if arrow_idx == -1:
                    print(f"Warning: No '->' found in line: {line}")
                    continue
                
                # Contraction is from parts[1] to parts[arrow_idx] inclusive
                contraction = ','.join(parts[1:arrow_idx + 1])
                
                # Dimension sizes are from arrow_idx+1 to second-to-last
                # Last value is arithmetic intensity
                dim_sizes = [int(x) for x in parts[arrow_idx + 1:-1]]
                ari_intensity = float(parts[-1])
                
                entries.append(BenchmarkEntry(
                    id=benchmark_id,
                    contraction=contraction,
                    dimension_sizes=dim_sizes,
                    arithmetic_intensity=ari_intensity
                ))
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line} - {e}")
                continue
    
    return entries


def infer_tensor_shapes(contraction: str, dim_sizes: list[int]) -> dict[str, tuple[int, ...]]:
    """
    Infer tensor shapes from the einsum contraction string and dimension sizes.
    
    The contraction string is like "abdfe,cf->abcde"
    We need to map each unique index to a dimension size.
    """
    # Parse the contraction string
    lhs, output = contraction.split('->')
    inputs = lhs.split(',')
    
    # Collect all unique indices in order of first appearance
    all_indices = []
    seen = set()
    for inp in inputs:
        for char in inp:
            if char not in seen:
                all_indices.append(char)
                seen.add(char)
    for char in output:
        if char not in seen:
            all_indices.append(char)
            seen.add(char)
    
    # Map indices to sizes
    if len(all_indices) != len(dim_sizes):
        raise ValueError(
            f"Mismatch: {len(all_indices)} indices ({all_indices}) "
            f"vs {len(dim_sizes)} dimension sizes"
        )
    
    index_to_size = {idx: size for idx, size in zip(all_indices, dim_sizes)}
    
    # Build shapes for each input tensor
    shapes = {}
    for i, inp in enumerate(inputs):
        shape = tuple(index_to_size[char] for char in inp)
        shapes[f"input_{i}"] = shape
    
    # Output shape
    shapes["output"] = tuple(index_to_size[char] for char in output)
    
    return shapes


def create_einsum_model(contraction: str, shapes: dict[str, tuple[int, ...]]) -> onnx.ModelProto:
    """
    Create an ONNX model with a single Einsum operation.
    """
    lhs, output = contraction.split('->')
    inputs = lhs.split(',')
    
    # Create input tensors
    input_tensors = []
    for i, inp in enumerate(inputs):
        shape = shapes[f"input_{i}"]
        input_tensors.append(
            helper.make_tensor_value_info(f"input_{i}", TensorProto.FLOAT, list(shape))
        )
    
    # Create output tensor
    output_shape = shapes["output"]
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output_shape))
    
    # Create the Einsum node
    einsum_node = helper.make_node(
        'Einsum',
        inputs=[f"input_{i}" for i in range(len(inputs))],
        outputs=['output'],
        equation=contraction
    )
    
    # Create the graph
    graph = helper.make_graph(
        [einsum_node],
        'einsum_graph',
        input_tensors,
        [output_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 8
    
    return model


def generate_random_inputs(shapes: dict[str, tuple[int, ...]], seed: int = 42) -> dict[str, np.ndarray]:
    """Generate random input tensors."""
    rng = np.random.default_rng(seed)
    inputs = {}
    for name, shape in shapes.items():
        if name.startswith("input_"):
            inputs[name] = rng.standard_normal(shape).astype(np.float32)
    return inputs


def benchmark_model(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
    warmup: int = 3,
    runs: int = 10,
    use_gpu: bool = False
) -> dict:
    """
    Benchmark an ONNX model using ONNX Runtime.
    
    Args:
        model: ONNX model to benchmark
        inputs: Input tensors
        warmup: Number of warmup iterations
        runs: Number of benchmark runs
        use_gpu: If True, use CUDA GPU backend; otherwise use CPU
    
    Returns timing statistics.
    """
    # Create session
    sess_options = ort.SessionOptions()
    
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        sess_options.intra_op_num_threads = 1  # Single-threaded for fair comparison
        sess_options.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options,
            providers=providers
        )
    except Exception as e:
        return {"error": str(e), "times": [], "mean": float('inf'), "std": 0.0}
    
    # Warmup
    for _ in range(warmup):
        try:
            session.run(None, inputs)
        except Exception as e:
            return {"error": str(e), "times": [], "mean": float('inf'), "std": 0.0}
    
    # Benchmark runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, inputs)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        "times": times,
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def run_single_benchmark(
    entry: BenchmarkEntry,
    warmup: int = 3,
    runs: int = 10,
    verify_correctness: bool = True,
    use_gpu: bool = False
) -> dict:
    """
    Run benchmark for a single einsum expression comparing all three approaches.
    
    Args:
        entry: Benchmark entry to run
        warmup: Number of warmup iterations
        runs: Number of benchmark runs
        verify_correctness: Whether to verify outputs match
        use_gpu: If True, use CUDA GPU backend; otherwise use CPU
    """
    result = {
        "id": entry.id,
        "contraction": entry.contraction,
        "dimension_sizes": entry.dimension_sizes,
        "arithmetic_intensity": entry.arithmetic_intensity,
    }
    
    try:
        # Infer shapes
        shapes = infer_tensor_shapes(entry.contraction, entry.dimension_sizes)
        result["shapes"] = {k: list(v) for k, v in shapes.items()}
        
        # Generate random inputs
        inputs = generate_random_inputs(shapes)
        
        # Select providers based on GPU flag
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # 1. Create and benchmark unoptimized einsum model
        print(f"  Creating unoptimized einsum model...")
        unopt_model = create_einsum_model(entry.contraction, shapes)
        print(f"  Benchmarking unoptimized model...")
        result["unoptimized"] = benchmark_model(unopt_model, inputs, warmup, runs, use_gpu)
        
        # Store reference output for correctness check
        if verify_correctness and "error" not in result["unoptimized"]:
            sess = ort.InferenceSession(
                unopt_model.SerializeToString(),
                providers=providers
            )
            reference_output = sess.run(None, inputs)[0]
        else:
            reference_output = None
        
        # 2. Binary tree without optimization
        print(f"  Creating binary tree (no optimization)...")
        try:
            tree_no_opt_model = lower_einsums(unopt_model, optimization_pass=False)
            result["tree_no_opt"] = benchmark_model(tree_no_opt_model, inputs, warmup, runs, use_gpu)
            
            # Verify correctness
            if verify_correctness and reference_output is not None and "error" not in result["tree_no_opt"]:
                sess = ort.InferenceSession(
                    tree_no_opt_model.SerializeToString(),
                    providers=providers
                )
                tree_output = sess.run(None, inputs)[0]
                result["tree_no_opt"]["correct"] = np.allclose(reference_output, tree_output, rtol=1e-4, atol=1e-4)
        except Exception as e:
            result["tree_no_opt"] = {"error": str(e)}
        
        # 3. Binary tree with optimization
        print(f"  Creating binary tree (with optimization)...")
        try:
            # Reload the model fresh for optimization pass
            tree_opt_model = lower_einsums(create_einsum_model(entry.contraction, shapes), optimization_pass=True)
            result["tree_optimized"] = benchmark_model(tree_opt_model, inputs, warmup, runs, use_gpu)
            
            # Verify correctness
            if verify_correctness and reference_output is not None and "error" not in result["tree_optimized"]:
                sess = ort.InferenceSession(
                    tree_opt_model.SerializeToString(),
                    providers=providers
                )
                tree_output = sess.run(None, inputs)[0]
                result["tree_optimized"]["correct"] = np.allclose(reference_output, tree_output, rtol=1e-4, atol=1e-4)
        except Exception as e:
            result["tree_optimized"] = {"error": str(e)}
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark einsum expressions with different optimization strategies"
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=str(Path(__file__).parent.parent / "benchmarks" / "single_node.txt"),
        help="Path to benchmark file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent / "benchmarks" / "results.json"),
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of benchmark IDs to run (e.g., '1,2,3'). Run all if not specified."
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip correctness verification"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (CUDA) backend instead of CPU"
    )
    
    args = parser.parse_args()
    
    # Parse benchmark file
    print(f"Loading benchmarks from: {args.benchmark_file}")
    entries = parse_benchmark_file(args.benchmark_file)
    print(f"Found {len(entries)} benchmark entries")
    
    # Filter by IDs if specified
    if args.ids:
        selected_ids = set(int(x) for x in args.ids.split(','))
        entries = [e for e in entries if e.id in selected_ids]
        print(f"Running {len(entries)} selected benchmarks: {sorted(e.id for e in entries)}")
    
    # Print backend info
    backend = "GPU (CUDA)" if args.gpu else "CPU"
    print(f"Backend: {backend}")
    
    # Run benchmarks
    results = []
    for i, entry in enumerate(entries):
        print(f"\n[{i+1}/{len(entries)}] Benchmark ID {entry.id}: {entry.contraction}")
        result = run_single_benchmark(
            entry,
            warmup=args.warmup,
            runs=args.runs,
            verify_correctness=not args.no_verify,
            use_gpu=args.gpu
        )
        results.append(result)
        
        # Print summary
        if "error" not in result:
            unopt_mean = result.get("unoptimized", {}).get("mean", float('inf'))
            tree_no_opt_mean = result.get("tree_no_opt", {}).get("mean", float('inf'))
            tree_opt_mean = result.get("tree_optimized", {}).get("mean", float('inf'))
            
            print(f"  Unoptimized:      {unopt_mean*1000:.3f} ms")
            print(f"  Tree (no opt):    {tree_no_opt_mean*1000:.3f} ms")
            print(f"  Tree (optimized): {tree_opt_mean*1000:.3f} ms")
            
            if unopt_mean > 0:
                speedup_no_opt = unopt_mean / tree_no_opt_mean if tree_no_opt_mean > 0 else 0
                speedup_opt = unopt_mean / tree_opt_mean if tree_opt_mean > 0 else 0
                print(f"  Speedup (no opt):    {speedup_no_opt:.2f}x")
                print(f"  Speedup (optimized): {speedup_opt:.2f}x")
        else:
            print(f"  Error: {result['error']}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "config": {
                "warmup": args.warmup,
                "runs": args.runs,
                "benchmark_file": args.benchmark_file,
                "backend": "GPU (CUDA)" if args.gpu else "CPU",
            },
            "results": results
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'ID':>4} {'Contraction':<25} {'Unopt (ms)':>12} {'NoOpt (ms)':>12} {'Opt (ms)':>12} {'Speedup':>8}")
    print("-"*80)
    
    for r in results:
        if "error" not in r:
            unopt = r.get("unoptimized", {}).get("mean", float('inf')) * 1000
            no_opt = r.get("tree_no_opt", {}).get("mean", float('inf')) * 1000
            opt = r.get("tree_optimized", {}).get("mean", float('inf')) * 1000
            speedup = unopt / opt if opt > 0 else 0
            
            contraction = r["contraction"][:24]
            print(f"{r['id']:>4} {contraction:<25} {unopt:>12.3f} {no_opt:>12.3f} {opt:>12.3f} {speedup:>7.2f}x")
        else:
            print(f"{r['id']:>4} ERROR: {r.get('error', 'Unknown')[:60]}")


if __name__ == "__main__":
    main()
