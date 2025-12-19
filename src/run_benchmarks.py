#!/usr/bin/env python3
"""
Benchmark script comparing four approaches for einsum operations:
1. Unoptimized einsum expressions (direct ONNX Einsum op)
2. Binary contraction tree without data layout optimization
3. Binary contraction tree with data layout optimization
4. XLA-compiled HLO from the optimized tree (optional)

Results are dumped to a JSON file for later analysis.
"""

import argparse
import json
import time
import subprocess
import tempfile
import os
from pathlib import Path
from typing import NamedTuple
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

from main import lower_einsums

# Path to XLA run_hlo_module binary
XLA_RUN_HLO_MODULE = "/workspace/xla/bazel-bin/xla/tools/run_hlo_module"


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


def onnx_to_mhlo(model: onnx.ModelProto, shapes: dict[str, tuple[int, ...]]) -> tuple[str, str]:
    """
    Convert an ONNX model with Einsum/MatMul operations to MHLO text format.
    
    Uses MHLO's native einsum operation which XLA will lower to dot_general.
    
    Returns:
        Tuple of (mhlo_text, format) where format is "mhlo"
    """
    mhlo_lines = []
    
    graph = model.graph
    
    # Collect input shapes and build shape map
    shape_map = {}
    input_types = []
    input_names_set = set(inp.name for inp in graph.input)
    
    for i, inp in enumerate(graph.input):
        shape = shapes.get(inp.name, None)
        if shape is None:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            shape = tuple(dims)
        shape_map[inp.name] = shape
        shape_str = "x".join(str(d) for d in shape)
        input_types.append(f"tensor<{shape_str}xf32>")
    
    # Get final output shape
    output_shape = shapes.get("output", None)
    if output_shape is None and graph.output:
        dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
        output_shape = tuple(dims)
    output_shape_str = "x".join(str(d) for d in output_shape) if output_shape else ""
    output_type = f"tensor<{output_shape_str}xf32>"
    
    # Build function signature
    arg_list = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(input_types))
    mhlo_lines.append(f"func.func @main({arg_list}) -> {output_type} {{")
    
    # Process each node
    var_counter = 0
    var_map = {inp.name: f"%arg{i}" for i, inp in enumerate(graph.input)}
    type_map = {inp.name: input_types[i] for i, inp in enumerate(graph.input)}
    last_var = None
    
    for node in graph.node:
        if node.op_type == "Einsum":
            equation = None
            for attr in node.attribute:
                if attr.name == "equation":
                    equation = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                    break
            
            if equation and len(node.input) == 2:
                lhs_name = var_map.get(node.input[0], node.input[0])
                rhs_name = var_map.get(node.input[1], node.input[1])
                
                lhs_shape = shape_map.get(node.input[0], ())
                rhs_shape = shape_map.get(node.input[1], ())
                
                lhs_type = type_map.get(node.input[0], f"tensor<{'x'.join(str(d) for d in lhs_shape)}xf32>")
                rhs_type = type_map.get(node.input[1], f"tensor<{'x'.join(str(d) for d in rhs_shape)}xf32>")
                
                # Parse einsum to compute output shape
                lhs_indices, rest = equation.split(",", 1)
                rhs_indices, out_indices = rest.split("->")
                
                # Build dimension sizes map
                dim_sizes = {}
                for i, c in enumerate(lhs_indices):
                    if i < len(lhs_shape):
                        dim_sizes[c] = lhs_shape[i]
                for i, c in enumerate(rhs_indices):
                    if i < len(rhs_shape):
                        dim_sizes[c] = rhs_shape[i]
                
                # Compute output shape from output indices
                result_shape = tuple(dim_sizes.get(c, 1) for c in out_indices)
                result_shape_str = "x".join(str(d) for d in result_shape)
                result_type = f"tensor<{result_shape_str}xf32>"
                
                var_name = f"%{var_counter}"
                var_counter += 1
                
                # Use MHLO's native einsum operation
                mhlo_lines.append(f'  {var_name} = "mhlo.einsum"({lhs_name}, {rhs_name}) {{einsum_config = "{equation}"}} : ({lhs_type}, {rhs_type}) -> {result_type}')
                
                shape_map[node.output[0]] = result_shape
                var_map[node.output[0]] = var_name
                type_map[node.output[0]] = result_type
                last_var = var_name
            elif equation and len(node.input) == 1:
                # Single-input einsum is a transpose/permutation
                # e.g., "abdfe->abfde" 
                input_name = var_map.get(node.input[0], node.input[0])
                input_shape = shape_map.get(node.input[0], ())
                input_type = type_map.get(node.input[0], f"tensor<{'x'.join(str(d) for d in input_shape)}xf32>")
                
                # Parse equation: "indices->out_indices"
                in_indices, out_indices = equation.split("->")
                
                # Build dimension sizes map
                dim_sizes = {}
                for i, c in enumerate(in_indices):
                    if i < len(input_shape):
                        dim_sizes[c] = input_shape[i]
                
                # Compute permutation: for each output index, find its position in input
                perm = [in_indices.index(c) for c in out_indices]
                
                # Compute output shape
                result_shape = tuple(dim_sizes.get(c, 1) for c in out_indices)
                result_shape_str = "x".join(str(d) for d in result_shape)
                result_type = f"tensor<{result_shape_str}xf32>"
                
                var_name = f"%{var_counter}"
                var_counter += 1
                
                perm_str = ", ".join(str(p) for p in perm)
                mhlo_lines.append(f'  {var_name} = "mhlo.transpose"({input_name}) {{permutation = dense<[{perm_str}]> : tensor<{len(perm)}xi64>}} : ({input_type}) -> {result_type}')
                
                shape_map[node.output[0]] = result_shape
                var_map[node.output[0]] = var_name
                type_map[node.output[0]] = result_type
                last_var = var_name
            else:
                return None, None
                
        elif node.op_type == "Transpose":
            input_name = var_map.get(node.input[0], node.input[0])
            input_shape = shape_map.get(node.input[0], ())
            
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
            
            if perm and input_shape:
                transposed_shape = [input_shape[p] for p in perm]
                transposed_shape_str = "x".join(str(d) for d in transposed_shape)
                result_type = f"tensor<{transposed_shape_str}xf32>"
                input_type = type_map.get(node.input[0], f"tensor<{'x'.join(str(d) for d in input_shape)}xf32>")
                
                var_name = f"%{var_counter}"
                var_counter += 1
                perm_str = ", ".join(str(p) for p in perm)
                mhlo_lines.append(f'  {var_name} = "mhlo.transpose"({input_name}) {{permutation = dense<[{perm_str}]> : tensor<{len(perm)}xi64>}} : ({input_type}) -> {result_type}')
                
                shape_map[node.output[0]] = tuple(transposed_shape)
                var_map[node.output[0]] = var_name
                type_map[node.output[0]] = result_type
                last_var = var_name
                
        elif node.op_type == "MatMul":
            lhs_name = var_map.get(node.input[0], node.input[0])
            rhs_name = var_map.get(node.input[1], node.input[1])
            lhs_shape = shape_map.get(node.input[0], ())
            rhs_shape = shape_map.get(node.input[1], ())
            
            if len(lhs_shape) < 2 or len(rhs_shape) < 2:
                return None, None
            
            lhs_type = type_map.get(node.input[0], f"tensor<{'x'.join(str(d) for d in lhs_shape)}xf32>")
            rhs_type = type_map.get(node.input[1], f"tensor<{'x'.join(str(d) for d in rhs_shape)}xf32>")
            
            # Construct einsum equation for MatMul
            lhs_rank = len(lhs_shape)
            
            batch_indices = "".join(chr(ord('a') + i) for i in range(lhs_rank - 2))
            i_idx = chr(ord('a') + lhs_rank - 2)
            j_idx = chr(ord('a') + lhs_rank - 1)
            k_idx = chr(ord('a') + lhs_rank)
            
            lhs_eq = batch_indices + i_idx + j_idx
            rhs_eq = batch_indices + j_idx + k_idx
            out_eq = batch_indices + i_idx + k_idx
            equation = f"{lhs_eq},{rhs_eq}->{out_eq}"
            
            # Output shape: batch + lhs[:-1] + rhs[-1]
            out_shape = list(lhs_shape[:-1]) + [rhs_shape[-1]]
            out_shape_str = "x".join(str(d) for d in out_shape)
            result_type = f"tensor<{out_shape_str}xf32>"
            
            var_name = f"%{var_counter}"
            var_counter += 1
            
            mhlo_lines.append(f'  {var_name} = "mhlo.einsum"({lhs_name}, {rhs_name}) {{einsum_config = "{equation}"}} : ({lhs_type}, {rhs_type}) -> {result_type}')
            
            shape_map[node.output[0]] = tuple(out_shape)
            var_map[node.output[0]] = var_name
            type_map[node.output[0]] = result_type
            last_var = var_name
    
    # Add return
    if last_var:
        final_output = graph.output[0].name if graph.output else None
        if final_output and final_output in var_map:
            last_var = var_map[final_output]
        mhlo_lines.append(f"  func.return {last_var} : {output_type}")
    else:
        mhlo_lines.append(f"  func.return %arg0 : {output_type}")
    
    mhlo_lines.append("}")
    
    return "\n".join(mhlo_lines), "mhlo"


def benchmark_xla(
    hlo_text: str,
    input_format: str = "hlo",
    warmup: int = 3,
    runs: int = 10,
    use_gpu: bool = False
) -> dict:
    """
    Benchmark an HLO/MHLO module using XLA's run_hlo_module tool.
    
    Parses timing from run_hlo_module output which reports "compiled and ran in Xs".
    Times include compilation.
    
    Args:
        hlo_text: The HLO or MHLO text
        input_format: Format of the input ("hlo", "mhlo", "stablehlo")
        warmup: Number of warmup runs
        runs: Number of benchmark runs
        use_gpu: Whether to use GPU
    """
    import re
    
    if not os.path.exists(XLA_RUN_HLO_MODULE):
        return {"error": f"XLA run_hlo_module not found at {XLA_RUN_HLO_MODULE}"}
    
    platform = "CUDA" if use_gpu else "CPU"
    
    # Write HLO to a temp file with appropriate suffix
    suffix = ".mhlo" if input_format == "mhlo" else ".hlo"
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(hlo_text)
        hlo_path = f.name
    
    try:
        # Use --iterations to run warmup + benchmark in a single invocation
        # This is more efficient and the timing is parsed from output
        total_iterations = warmup + runs
        
        base_args = [
            XLA_RUN_HLO_MODULE,
            f"--platform={platform}", 
            f"--input_format={input_format}",
            "--reference_platform=",  # Skip interpreter reference
            "--run_test_hlo_passes=true",
            f"--iterations={total_iterations}",
            hlo_path
        ]
        
        result = subprocess.run(
            base_args,
            capture_output=True,
            text=True,
            timeout=300  # Longer timeout for multiple iterations
        )
        
        if result.returncode != 0:
            return {"error": f"XLA run failed: {result.stderr[:500]}"}
        
        # Parse timing from output: "... compiled and ran in 0.123456s."
        pattern = r"compiled and ran in ([\d.]+)s"
        matches = re.findall(pattern, result.stdout + result.stderr)
        
        if len(matches) < total_iterations:
            return {"error": f"Could not parse timing from XLA output. Got {len(matches)} times, expected {total_iterations}"}
        
        # Skip warmup iterations, take only benchmark runs
        all_times = [float(t) for t in matches]
        times = all_times[warmup:warmup + runs]
        
        return {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
        }
    except subprocess.TimeoutExpired:
        return {"error": "XLA execution timed out"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(hlo_path)


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
        print(f"  Benchmarking unoptimized model (ONNX Runtime)...")
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
        tree_no_opt_model = None
        try:
            tree_no_opt_model = lower_einsums(unopt_model, optimization_pass=False)
            print(f"  Benchmarking tree (no opt) model (ONNX Runtime)...")
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
        
        # XLA benchmark for tree_no_opt model
        print(f"  Benchmarking tree (no opt) model (XLA)...")
        try:
            if tree_no_opt_model is not None:
                mhlo_text, input_format = onnx_to_mhlo(tree_no_opt_model, shapes)
                if mhlo_text:
                    result["tree_no_opt_xla"] = benchmark_xla(mhlo_text, input_format, warmup, runs, use_gpu)
                else:
                    result["tree_no_opt_xla"] = {"error": "Failed to convert ONNX to MHLO"}
            else:
                result["tree_no_opt_xla"] = {"error": "No tree_no_opt model available"}
        except Exception as e:
            result["tree_no_opt_xla"] = {"error": str(e)}
        
        # 3. Binary tree with optimization
        print(f"  Creating binary tree (with optimization)...")
        tree_opt_model = None
        try:
            # Reload the model fresh for optimization pass
            tree_opt_model = lower_einsums(create_einsum_model(entry.contraction, shapes), optimization_pass=True)
            print(f"  Benchmarking tree (optimized) model (ONNX Runtime)...")
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
        
        # XLA benchmark for tree_optimized model
        print(f"  Benchmarking tree (optimized) model (XLA)...")
        try:
            if tree_opt_model is not None:
                mhlo_text, input_format = onnx_to_mhlo(tree_opt_model, shapes)
                if mhlo_text:
                    result["tree_optimized_xla"] = benchmark_xla(mhlo_text, input_format, warmup, runs, use_gpu)
                else:
                    result["tree_optimized_xla"] = {"error": "Failed to convert ONNX to MHLO"}
            else:
                result["tree_optimized_xla"] = {"error": "No optimized model available"}
        except Exception as e:
            result["tree_optimized_xla"] = {"error": str(e)}
        
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
            # ONNX Runtime results
            unopt_mean = result.get("unoptimized", {}).get("mean", float('inf'))
            tree_no_opt_mean = result.get("tree_no_opt", {}).get("mean", float('inf'))
            tree_opt_mean = result.get("tree_optimized", {}).get("mean", float('inf'))
            
            # XLA results
            tree_no_opt_xla = result.get("tree_no_opt_xla", {})
            tree_opt_xla = result.get("tree_optimized_xla", {})
            
            print(f"  === ONNX Runtime ===")
            print(f"  Unoptimized:      {unopt_mean*1000:.3f} ms")
            print(f"  Tree (no opt):    {tree_no_opt_mean*1000:.3f} ms")
            print(f"  Tree (optimized): {tree_opt_mean*1000:.3f} ms")
            
            print(f"  === XLA (tree models only) ===")
            if "error" in tree_no_opt_xla:
                print(f"  Tree (no opt):    ERROR - {tree_no_opt_xla['error'][:40]}")
            else:
                print(f"  Tree (no opt):    {tree_no_opt_xla.get('mean', 0)*1000:.3f} ms")
            if "error" in tree_opt_xla:
                print(f"  Tree (optimized): ERROR - {tree_opt_xla['error'][:40]}")
            else:
                print(f"  Tree (optimized): {tree_opt_xla.get('mean', 0)*1000:.3f} ms")
            
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
    
    # Print summary table - ONNX Runtime
    print("\n" + "="*120)
    print("SUMMARY - ONNX Runtime")
    print("="*120)
    print(f"{'ID':>4} {'Contraction':<25} {'Unopt (ms)':>12} {'NoOpt (ms)':>12} {'Opt (ms)':>12} {'Speedup':>8}")
    print("-"*75)
    
    for r in results:
        if "error" not in r:
            unopt = r.get("unoptimized", {}).get("mean", float('inf')) * 1000
            no_opt = r.get("tree_no_opt", {}).get("mean", float('inf')) * 1000
            opt = r.get("tree_optimized", {}).get("mean", float('inf')) * 1000
            speedup = unopt / opt if opt > 0 else 0
            
            contraction = r["contraction"][:24]
            print(f"{r['id']:>4} {contraction:<25} {unopt:>12.3f} {no_opt:>12.3f} {opt:>12.3f} {speedup:>7.2f}x")
        else:
            print(f"{r['id']:>4} ERROR: {r.get('error', 'Unknown')[:70]}")
    
    # Print summary table - XLA (tree models only)
    print("\n" + "="*100)
    print("SUMMARY - XLA (tree models only)")
    print("="*100)
    print(f"{'ID':>4} {'Contraction':<25} {'NoOpt (ms)':>12} {'Opt (ms)':>12} {'Speedup':>8}")
    print("-"*65)
    
    for r in results:
        if "error" not in r:
            no_opt_xla = r.get("tree_no_opt_xla", {})
            opt_xla = r.get("tree_optimized_xla", {})
            
            no_opt = no_opt_xla.get("mean", float('inf')) * 1000 if "error" not in no_opt_xla else float('inf')
            opt = opt_xla.get("mean", float('inf')) * 1000 if "error" not in opt_xla else float('inf')
            speedup = no_opt / opt if opt > 0 and opt != float('inf') and no_opt != float('inf') else 0
            
            contraction = r["contraction"][:24]
            no_opt_str = f"{no_opt:>12.3f}" if no_opt != float('inf') else "       ERROR"
            opt_str = f"{opt:>12.3f}" if opt != float('inf') else "       ERROR"
            speedup_str = f"{speedup:>7.2f}x" if speedup > 0 else "     N/A"
            print(f"{r['id']:>4} {contraction:<25} {no_opt_str} {opt_str} {speedup_str}")
        else:
            print(f"{r['id']:>4} ERROR: {r.get('error', 'Unknown')[:70]}")


if __name__ == "__main__":
    main()
