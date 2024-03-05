import os
from copy import deepcopy
from pathlib import Path
import logging

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
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.get_input import InputGenerator, get_cf_args, get_dummy_input
from chop.tools.utils import parse_accelerator, to_numpy_if_tensor

from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
import pprint
import pdb

import gc
gc.collect()

import torch.nn.utils.prune as prune

global act_masks
act_masks = None

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)

def pre_transform_load(mask, load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl"]:
        model = load_model(mask, load_name=load_name, load_type=load_type, model=model)
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
    #pdb.set_trace()
    load_name = config['retrain']['load_name']
    load_type = config['retrain']['load_type']
    
    mask=None
    model = pre_transform_load(mask, load_name=load_name, load_type=load_type, model=model)
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
        match pass_name:
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
                print("pass_config: ") ; print(pass_config)
                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                #prune_save_dir = save_dir / "prune"
                #prune_save_dir.mkdir(parents=True, exist_ok=True)
                #prune_save_dir没有任何内容
                batch_size = config['retrain']['training']['batch_size']
                graph, _ = PASSES[pass_name](
                    graph,
                    batch_size, # self_added
                    pass_config,
                )
                graph, sparsity_info, mask_collect, act_masks = PASSES["add_pruning_metadata"](
                    graph,
                    {"dummy_in": dummy_in, "add_value": False}
                )
                torch.save(act_masks, "/mnt/d/imperial/second_term/adls/projects/mase/machop/act_masks.pth")
                print("activation mask saved")
                #pdb.set_trace()
                pp.pprint(sparsity_info)
                del act_masks

            case "quantize":
                #pdb.set_trace()
                gc.collect()
                pass_save_dir = save_dir / "quantize"
                graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
                ori_graph = deepcopy_mase_graph(graph)
                #ori_graph = deepcopy(graph)
                graph, _ = PASSES["quantize"](graph, pass_args=pass_config)
                PASSES["summarize_quantization"](ori_graph, graph, save_dir=pass_save_dir)

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


    ###############################
    #re-train
    ###############################
        
    from pytorch_lightning.callbacks import Callback

    class HessianComputationCallback(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # 这里仅作为示例，只选择第一个权重参数来计算 Hessian 的对角线
            loss = outputs['loss']
            named_parameters = list(pl_module.named_parameters())
            name, param = named_parameters[1]
            pdb.set_trace()
            if 'weight' in name:
                hessian_diag = self.compute_hessian_diag(param, pl_module, loss)
                # 打印 Hessian 对角线的统计摘要
                print(f"[Batch {batch_idx}] Hessian Diagonal for {name}: max={hessian_diag.max().item()}, min={hessian_diag.min().item()}, mean={hessian_diag.mean().item()}")
        @staticmethod
        def compute_hessian_diag(param, model, loss):
            model.eval()
            first_order_grads = torch.autograd.grad(loss, param, create_graph=True, allow_unused=True)

            hessian_diag = []
            for grad in first_order_grads:
                if grad is not None:
                    grad_grad = torch.autograd.grad(grad, param, retain_graph=True)[0]
                    hessian_diag.append(grad_grad)

            hessian_diag = torch.stack(hessian_diag).view_as(param)
            return hessian_diag

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
        hessian_callback = HessianComputationCallback()
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        plt_trainer_args["callbacks"] = [
            checkpoint_callback,
            #hessian_callback,
            lr_monitor_callback,
        ]
        plt_trainer_args["logger"] = visualizer

    plugins = None
    plt_trainer_args["plugins"] = plugins


    wrapper_cls = get_model_wrapper(model_info, task)

    load_name = "/mnt/d/imperial/second_term/adls/projects/mase/mase_output/vgg_cifar10_prune/software/prune/transformed_ckpt/state_dict.pt"
    load_type = "pt"
    #import pdb; pdb.set_trace()

    if load_name is not None:
        model = load_model(mask_collect, load_name, load_type=load_type, model=model)
        #model = load_model(load_name, load_type=load_type, model=model)
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