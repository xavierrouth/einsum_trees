import argparse
from pathlib import Path
import numpy as np

import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset15 as op
from onnxscript.onnx_types import FLOAT
import onnx_ir as ir
from codegen import lower_einsum_node  # custom optimization pass

def lower_einsums(model):
    """Custom optimization pass on ONNX model"""
    model = ir.from_proto(model)
    graph = model.graph
    
    # Example: Remove consecutive identity operations
    nodes_to_remove = []

    breakpoint()

    # ir_graph = ir.from_proto(graph)
    for i in range(graph.num_nodes()):
        node = graph.node(i)
        print(f"Node {i}: {node.op_type}")
        if node.op_type == 'Einsum':
            #check that inputs are stored in row major format
            breakpoint()
            # ir_node = ir.from_proto(node)
            lowered_node = lower_einsum_node(graph, model, node)
            # graph.node.remove(node)
            # graph.append(lowered_node)
            ir.convenience.replace_all_uses_with(node.outputs[0], lowered_node.outputs[0], replace_graph_outputs=True)
            graph.remove(node)

            breakpoint()
            # graph.add_node(lowered_node)
            # graph.node.insert(i, lowered_node)
            # nodes_to_remove.append(node)

        # breakpoint()
        # if node.op_type == 'Identity':
        #     # Find if output feeds into another Identity
        #     for j, next_node in enumerate(graph.node[i+1:], i+1):
        #         if next_node.op_type == 'Identity' and next_node.input[0] == node.output[0]:
        #             nodes_to_remove.append(next_node)
    
    # Remove marked nodes
    # for node in nodes_to_remove:
    #     graph.node.remove(node)
    
    return ir.to_proto(model)


def main():
    p = argparse.ArgumentParser(description="Simple ONNX model optimizer")
    p.add_argument("--input-model", type=str, required=True, help="Path to input ONNX model")
    args = p.parse_args()

    input_path = Path(args.input_model)
    if not input_path.is_file():
        raise SystemExit(f"Input model not found: {input_path}")

    model = onnx.load(input_path.as_posix())

    # Rename graph outputs
    # for out in model.graph.output:
    #     out.name = f"optimized_{out.name}"
    
    optimized_model = lower_einsums(model)

    output_path = input_path.with_name(f"optimized_{input_path.name}")
    onnx.save(optimized_model, output_path.as_posix())

    print(f"Loaded model: {input_path}")
    print(f"Saved optimized model: {output_path}")


if __name__ == "__main__":
    main()

