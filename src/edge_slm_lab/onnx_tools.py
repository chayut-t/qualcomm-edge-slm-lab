"""Read an ONNX file and report its interface contract (Task 04 teaching code).

An ONNX graph declares its interface up front: named inputs and outputs,
each with a dtype and a shape. This module turns that declaration into
plain dictionaries, so a notebook can print it, assert on it, and save it
as JSON. Nothing here runs the model.
"""

from __future__ import annotations

import os
from collections import Counter

import onnx


def _tensor_info(value_info) -> dict:
    """Name, dtype, and shape of one graph input or output."""
    ttype = value_info.type.tensor_type
    dims = []
    for d in ttype.shape.dim:
        # dim_value > 0 means a fixed size; anything else is dynamic.
        dims.append(d.dim_value if d.dim_value > 0 else d.dim_param or None)
    return {
        "name": value_info.name,
        "dtype": onnx.TensorProto.DataType.Name(ttype.elem_type).lower(),
        "shape": dims,
    }


def graph_io(path) -> dict:
    """The declared contract: {"inputs": [...], "outputs": [...]}.

    Weights are stored as initializers, not inputs, so "inputs" lists
    only the tensors a caller must supply.
    """
    model = onnx.load(str(path))
    return {
        "inputs": [_tensor_info(v) for v in model.graph.input],
        "outputs": [_tensor_info(v) for v in model.graph.output],
    }


def op_counts(path) -> dict:
    """How many nodes of each op type the graph contains, most common first."""
    model = onnx.load(str(path))
    return dict(Counter(node.op_type for node in model.graph.node).most_common())


def all_shapes_fixed(io: dict) -> bool:
    """True when every declared input and output dimension is a fixed integer."""
    tensors = io["inputs"] + io["outputs"]
    return all(isinstance(d, int) for t in tensors for d in t["shape"])


def contract(path, graph_name: str) -> dict:
    """One JSON-ready record: the full interface plus size and node stats."""
    model = onnx.load(str(path))
    io = graph_io(path)
    return {
        "graph": graph_name,
        "file": os.path.basename(str(path)),
        "size_bytes": os.path.getsize(str(path)),
        "opset": max(op.version for op in model.opset_import),
        "num_nodes": len(model.graph.node),
        "num_initializers": len(model.graph.initializer),
        "all_shapes_fixed": all_shapes_fixed(io),
        **io,
        "op_counts": op_counts(path),
    }
