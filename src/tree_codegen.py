import onnx
import onnxruntime as ort
from onnxscript import script
from onnxscript import opset15 as op
from onnx import NodeProto
import onnx_ir as ir
import numpy as np

from einsum_tree import contraction, TreeNode, optimize_tree

def lower_einsum_node(graph: ir.Graph, model, einsum_node: ir.Node, optimize=False) -> ir.Node:
    # dim_sizes = get_dimension_sizes(model, einsum_node)
    # Get dimension sizes
    # breakpoint()
    # parse the einsum string
    if einsum_node.op_type != "Einsum":
        raise ValueError("Input node is not an Einsum node")
    # breakpoint()
    einsum_string = einsum_node.attributes['equation'].value

    tree_node: TreeNode = contraction(einsum_string)
    # tree_node.print_tree()
    if optimize:
        tree_node = optimize_tree(tree_node)
        # tree_node.print_tree()

    # breakpoint()

    # Map leaves to original inputs by NAME, and matching location in einsum string

    inputs = einsum_string.split("->")[0].split(",")
    # Map inputs to their input number
    value_map = {inputs[i]: einsum_node.inputs[i] for i in range(len(inputs))}

    # breakpoint()
    def lower(node, graph):
        nonlocal value_map
        # breakpoint()

        if node is None:
            return None
        name = node.value_name()

        if name in value_map:
            return value_map[name]
        
        # Lower children! 

        # left first
        child_values = []
        lhs_node = lower(node.lhs, graph)
        if lhs_node is not None:
            child_values.append(lhs_node)
        rhs_node = lower(node.rhs, graph)
        if rhs_node is not None:
            child_values.append(rhs_node)
        
        # Create the einsum node
        einsum_eq = node.einsum_string
        new_node = ir.node(
            graph=graph,
            op_type="Einsum",
            inputs=child_values,
            attributes={
                "equation": einsum_eq
            }
        )

        value_map[name] = new_node.outputs[0]
        return new_node.outputs[0]

    # Might need to get the NODE of this output value, not the output value.
    final_output_value = lower(tree_node, graph)
    return final_output_value.producer()
    # tree_node.