import logging

import toml
import torch
import torch.fx as fx
from chop.ir.graph.mase_metadata import MaseMetadata
from chop.passes.graph.analysis.utils import (
    get_input_nodes,
    get_output_nodes,
)
from torch import nn

from .hardware_metadata_layers import (
    analyse_hardware_parameters_linear,
    analyse_hardware_parameters_relu,
    analyse_hardware_parameters_batch_norm1d,
    analyse_hardware_parameters_custom_layer,
    analyse_hardware_parameters_layer_norm,
)

logger = logging.getLogger(__name__)


def analysis_hardware_parameters(node):
    if node.meta["mase"].parameters["hardware"]["is_implicit"]:
        return
    op = node.meta["mase"].parameters["common"]["mase_op"]

    if op == "linear":
        node.meta["mase"] = analyse_hardware_parameters_linear(node.meta["mase"])
    elif op == "relu":
        node.meta["mase"] = analyse_hardware_parameters_relu(node.meta["mase"])
    elif op == "batch_norm1d":
        node.meta["mase"] = analyse_hardware_parameters_batch_norm1d(node.meta["mase"])
    elif op == "patched_custom_layers":
        node.meta["mase"] = analyse_hardware_parameters_custom_layer(node.meta["mase"])
    elif op == "layer_norm":
        node.meta["mase"] = analyse_hardware_parameters_layer_norm(node.meta["mase"])
    else:
        # Implicit functions: getitem/getattr, assert, size, reshape etc
        node.meta["mase"].parameters["hardware"]["is_implicit"] = True
        # raise ValueError(f"Unknown mase op: {op}")


def add_hardware_metadata_analysis_pass(graph, pass_args=None):
    """add hardware metadata
    This is a standard analysis pass that runs at the start of all transform calls

    name_style_pass (graph, pass_args)

    This follows the the naming convention of
    [name]_[style]_pass
    add_hardware_metadata(name)_analysis(style)_pass

    passname : {args}

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass does not need any arguments, defaults to None
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)
    """
    for node in graph.fx_graph.nodes:
        node.meta["mase"].parameters["hardware"]["is_implicit"] = False
    graph.nodes_in = get_input_nodes(graph.fx_graph)
    graph.nodes_out = get_output_nodes(graph.fx_graph)

    for node in graph.fx_graph.nodes:
        analysis_hardware_parameters(node)
    return graph, {}