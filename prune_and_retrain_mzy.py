import os
from copy import deepcopy
from pathlib import Path
import logging
import pdb
import torch
from chop.passes.graph import PASSES

import pytorch_lightning as pl
from chop.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from chop.tools.get_input import get_dummy_input

from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
)


from chop.passes.graph.interface import save_mase_graph_interface_pass
from chop.ir.graph import MaseGraph
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from torch.distributed.fsdp import FullyShardedDataParallel
from pytorch_lightning.strategies import DDPStrategy
from chop.tools.config_load import load_config

from chop.tools.get_input import InputGenerator, get_cf_args, get_dummy_input


from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.interface import (
    load_mase_graph_interface_pass,
    save_mase_graph_interface_pass,
)
from chop.passes.graph.utils import deepcopy_mase_graph, get_node_actual_target
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.get_input import InputGenerator, get_cf_args, get_dummy_input
from chop.tools.utils import parse_accelerator, to_numpy_if_tensor

from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
import pprint

import gc
gc.collect()

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)

# 量化函数，将tensor值量化为8位整数
def quantize_tensor(tensor, num_bits=8):
    scale_factor = (2**(num_bits-1) - 1) / 0.5  # 0.5是一个适当的缩放因子，根据你的情况可能需要调整
    quantized_tensor = torch.round(tensor * scale_factor).to(torch.int8)
    return quantized_tensor


def pre_transform_load(load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl"]:
        model = load_model(load_name=load_name, load_type=load_type, model=model)
    return model


def prune_and_retrain(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    dataset_info,
    task,
    config,
    visualizer,
    prune_save_dir: str = None,
    retrain_save_path: str=None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    accelerator = parse_accelerator(accelerator)
    config = load_config(config)

    model = pre_transform_load(load_name=load_name, load_type=load_type, model=model)
    model.to(accelerator)
    
    save_dir = prune_save_dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # concrete forward args for freezing dynamic control flow in forward pass
    if "cf_args" not in config:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)
    else:
        cf_args = config["cf_args"]

    # graph generation
    graph = MaseGraph(model=model, cf_args=cf_args)
    # graph_metadata = Mase
    graph, _ = init_metadata_analysis_pass(graph, pass_args=None)
    # logger.info(f"graph: {graph.fx_graph}")

    # create or load metadata.parameters and mase_graph.model
    if load_name is not None and load_type == "mz":
        graph, _ = load_mase_graph_interface_pass(graph, pass_args=load_name)
    else:
        dummy_in = get_dummy_input(
            model_info=model_info,
            data_module=data_module,
            task=task,
            device=accelerator,
        )
        if len(graph.model.additional_inputs) > 0:
            dummy_in = dummy_in | graph.model.additional_inputs
        graph, _ = add_common_metadata_analysis_pass(
            graph, pass_args={"dummy_in": dummy_in}
        )
        graph, _ = add_software_metadata_analysis_pass(graph, pass_args=None)

    pass_config = config["passes"]

    for pass_name, pass_config in pass_config.items():
        pass_name: str
        pass_config: dict
        # import pdb; pdb.set_trace()
        match pass_name:
            case "quantize":
                pass_save_dir = save_dir / "quantize"
                graph, _ = metadata_value_type_cast_transform_pass(
                    graph, pass_args={"fn": to_numpy_if_tensor}
                )
                ori_graph = deepcopy_mase_graph(graph)
                graph, _ = PASSES["quantize"](graph, pass_args=pass_config)
                PASSES["summarize_quantization"](
                    ori_graph, graph, save_dir=pass_save_dir
                )
            case "prune":
                # NOTE: The input generator is only used for when the user wants to
                # enforce or observe activation sparsity. Otherwise, it's ignored.
                # We use the validation dataloader as that doesn't shuffle the input
                # data. This determinism helps establish a fair ground in draw
                # layer-wise comparisons between activation pruning strategies.
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="val",
                )
                print("pass_config") ; print(pass_config)
                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                # pass_config["by"] = "type"
                # pass_config["default"]["config"]["name"] = None
                #prune_save_dir = save_dir / "prune"
                #prune_save_dir.mkdir(parents=True, exist_ok=True)
                #prune_save_dir没有任何内容
                graph, _ = PASSES[pass_name](
                    graph,
                    #save_dir=prune_save_dir,
                    pass_config,
                )
                graph, sparsity_info, mask_collect = PASSES["add_pruning_metadata"](
                    graph,
                    {"dummy_in": dummy_in, "add_value": False}
                )
                pp.pprint(sparsity_info)
                # import pdb; pdb.set_trace()
            case "remove_prune_wrappers":
                # Removes the pruning-related hooks and makes pruning permanent
                graph, _ = PASSES[pass_name](graph, pass_args=None)

        assert isinstance(
            graph, MaseGraph
        ), f"Return type of {pass_name} must be MaseGraph, got {type(graph)}"

    if save_dir is not None:
        #import pdb;pdb.set_trace()
        transformed_ckpt = save_dir / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        graph, _ = metadata_value_type_cast_transform_pass(
            graph, pass_args={"fn": to_numpy_if_tensor}
        )
        graph, _ = save_mase_graph_interface_pass(graph, pass_args=transformed_ckpt) # save the pruned model

    for node in graph.fx_graph.nodes:
        target = get_node_actual_target(node)
        if hasattr(target, 'weight'):
            weight_value = target.weight
            # import pdb; pdb.set_trace()
            # for i in range(weight_value.shape[0]):
            #     for j in range(weight_value.shape[1]):
            #         # print(j)
            #         for k in range(weight_value.shape[2]):
            #             for n in range(weight_value.shape[3]):
            #                 if weight_value[i][j][k][n].item() == 0:
            #                     # print(weight_value[i][j][k][n])
            #                     weight_value[i][j][k][n] = quantize_tensor(weight_value[i][j][k][n])
            
            for i in range(weight_value.shape[0]):
                for j in range(3):
                    for k in range(3):
                        for n in range(3):
                            # print(f"Indices: ({i}, {j}, {k}, {n})")
                            # try:
                            value = weight_value[i][j][k][n].item()
                            if value == 0:
                                # Perform quantization pass
                                weight_value[i][j][k][n] = quantize_tensor(value)
                            # except IndexError:
                            #     print("Index out of bounds for weight_value")
                            #     print("Shape:", weight_value.shape)
                            #     print("Indices:", (i, j, k, n))
                                # raise


    ###############################
    #re-train
    ###############################

    plt_trainer_args={}
    if retrain_save_path is not None:
        # if retrain_save_path is None, the model will not be saved
        if not os.path.isdir(retrain_save_path):
            os.makedirs(retrain_save_path)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss_epoch",
            mode="min",
            filename="best",
            dirpath=retrain_save_path,
            save_last=True,
        )
        
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        plt_trainer_args["callbacks"] = [
            checkpoint_callback,
            lr_monitor_callback,
        ]
        plt_trainer_args["logger"] = visualizer

    plugins = None
    plt_trainer_args["plugins"] = plugins


    wrapper_cls = get_model_wrapper(model_info, task)

    # pdb.set_trace()
    load_name = config['retrain']['load_name']
    load_type = config['retrain']['load_type']
    #import pdb; pdb.set_trace()

    if load_name is not None:
        model = load_model(mask_collect, load_name, load_type=load_type, model=model)
        logger.info(f"'{load_type}' checkpoint loaded before training")

    plt_trainer_args['accelerator'] = config['retrain']['trainer']['accelerator']
    plt_trainer_args['devices'] = config['retrain']['trainer']['devices']

    pl_model = wrapper_cls(
        model,
        dataset_info=dataset_info,
        learning_rate = config['retrain']['training']['learning_rate'],
        epochs = config['retrain']['training']['max_epochs'],
        weight_decay = config['retrain']['training']['weight_decay'],
        optimizer = config['retrain']['training']['optimizer'],
        batch_size = config['retrain']['training']['batch_size'],
    )

    trainer = pl.Trainer(**plt_trainer_args, max_epochs=config['retrain']['training']['max_epochs'])

    trainer.fit(
        pl_model,
        datamodule=data_module,
    )
    # 训练的目的，不只是为了保存模型，更是为了去看准确率（也就是prune的效果到底有没有）

    '''
    if retrain_save_path is not None and load_name is not None and load_type == "mz":
        graph = MaseGraph(model)
        dummy_input = get_dummy_input(model_info, data_module, task)
        graph = init_metadata_analysis_pass(graph, None)
        graph = add_common_metadata_analysis_pass(graph, dummy_input)
        graph = add_software_metadata_analysis_pass(graph, None)
        train_ckpt = Path(retrain_save_path) / "train_ckpt"
        train_ckpt.mkdir(parents=True, exist_ok=True)
        save_mase_graph_interface_pass(graph, pass_args=train_ckpt)
    '''
