import json
import logging
from typing import Tuple

import torch
import torch.utils.data
from torch import nn as nn

import edges_removal.ugs_utils as ugs_utils
from common.data.modules import DataModule
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import Evaluator, TrainEvaluator, ComposeEvaluator
from common.experiment import FitExperimentBase, ExperimentResult
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from edges_removal.graph_model import GraphModel
from edges_removal.edge_removal_datamodule import EdgeRemovalDataModule
from edges_removal.utils import GNN_TYPE
from evaluation.torch_geometric_supervised_evaluator import TorchGeometricSupervisedValidationEvaluator, \
    TorchGeometricSupervisedTrainEvaluator
from train.torch_geometric_supervised_trainer import TorchGeometricSupervisedTrainer


class EdgesRemovalExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--model", default=GNN_TYPE.GCN, type=GNN_TYPE.from_string, choices=list(GNN_TYPE), required=False,
                            help="Which GNN model to use")

        parser.add_argument("--hidden_dim", default=32, type=int, required=False, help="Size of the hidden dimension")
        parser.add_argument("--num_layers", default=3, type=int, required=False, help="Number of layers to use (depth of network)")
        parser.add_argument("--train_fraction", default=0.8, type=float, required=False, help="Fraction of samples used for training")
        parser.add_argument("--val_fraction", default=0.1, type=float, required=False, help="Fraction of samples used for validation")
        parser.add_argument("--train_frac_seed", default=0.8, type=float, required=False, help="Seed for random split of the dataset")

        parser.add_argument("--no_layer_norm", action="store_true", help="Don't use layer norm if True")
        parser.add_argument("--load_dataset_to_gpu", action="store_true", help="Stores all dataset on the main GPU (if GPU device is given)")
        parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers to use")
        parser.add_argument("--batch_size", type=int, default=-1, help="Batch size. If < 0 (default) will use whole training set at each batch.")
        parser.add_argument("--accum_grad", default=1, type=int, required=False, help="Number of steps to accumulate gradients")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        parser.add_argument("--weight_decay", type=float, default=0, help="L2 regularization coefficient")
        parser.add_argument("--dataset_name", type=str, default='cora',
                            help="Graph dataset name, supported: cora, citeseer, pubmed, chameleon, crocodile, squirrel, dblp, cora_ml")
        parser.add_argument("--is_ugs_mask_train", type=bool, default=False, help="If true, learn ugs mask")
        parser.add_argument("--edges_ratio", type=float, default=1.0,
                            help="Ratio of edges being used from the graph. If different than 0 or 1, must supply edges_removal_conf")
        parser.add_argument("--edges_removal_conf", type=float, default=1.0, help="Configuration file for edges removal order")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None
        datamodule = EdgeRemovalDataModule(dataset_name=config["dataset_name"],
                                           train_fraction=config["train_fraction"],
                                           val_fraction=config["val_fraction"],
                                           batch_size=config["batch_size"],
                                           dataloader_num_workers=config["dataloader_num_workers"],
                                           train_frac_seed=config['train_fraction_seed'],
                                           edges_ratio=config["edges_ratio"],
                                           edges_remove_conf_file=config["edges_removal_conf"],
                                           load_dataset_to_device=load_dataset_to_device)
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: EdgeRemovalDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        num_layers = config["num_layers"]
        gnn_type = GNN_TYPE.from_string(config["model"])
        init_path = config['model_initialization_path'] if 'model_initialization_path' in config else None
        model = GraphModel(gnn_type=gnn_type, num_layers=num_layers, dim0=datamodule.dim0, h_dim=config["hidden_dim"],
                           out_dim=datamodule.out_dim, num_edges=datamodule.num_edges,
                           layer_norm=not config["no_layer_norm"],
                           is_ugs_mask_train=config['is_ugs_mask_train'],
                           model_initialization_path=init_path)
        return model

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: EdgeRemovalDataModule,
                                               val_dataloader: torch.utils.data.DataLoader, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [
            metrics.MetricInfo("train loss", metrics.CrossEntropyLoss(), tag="loss"),
            metrics.MetricInfo("train accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy")
        ]

        train_evaluator = TorchGeometricSupervisedTrainEvaluator(train_metric_info_seq)

        val_metric_info_seq = [
            metrics.MetricInfo("val loss", metrics.CrossEntropyLoss(), tag="loss"),
            metrics.MetricInfo("val accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy")
        ]

        val_evaluator = TorchGeometricSupervisedValidationEvaluator(model, val_dataloader, metric_info_seq=val_metric_info_seq, device=device)

        test_metric_info_seq = [
            metrics.MetricInfo("test loss", metrics.CrossEntropyLoss(), tag="loss"),
            metrics.MetricInfo("test accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy")
        ]

        test_dataloader = datamodule.test_dataloader()
        test_evaluator = TorchGeometricSupervisedValidationEvaluator(model, test_dataloader, metric_info_seq=test_metric_info_seq, device=device)
        return train_evaluator, ComposeEvaluator([val_evaluator, test_evaluator])

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train accuracy", is_train_metric=False, largest=True, return_best_score=False)

    def create_additional_metadata_to_log(self, model: nn.Module, datamodule: EdgeRemovalDataModule, config: dict, state: dict,
                                          logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["input dim"] = datamodule.dim0
        additional_metadata["output dim"] = datamodule.out_dim
        additional_metadata["train dataset size"] = len((datamodule.train_dataset.data.y >= 0).nonzero())
        additional_metadata["val dataset size"] = len((datamodule.val_data.y >= 0).nonzero())
        additional_metadata["test dataset size"] = len(((datamodule.test_data.y >= 0).nonzero()))
        return additional_metadata

    def create_trainer(self, model: nn.Module, datamodule: EdgeRemovalDataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config['is_ugs_mask_train']:
            reg_fn = ugs_utils.L1MaskRegulazation(config['ugs_s1'])
        else:
            reg_fn = None

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        return TorchGeometricSupervisedTrainer(model, optimizer, loss_fn, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                               callback=callback, reg_fn=reg_fn, gradient_accumulation=config["accum_grad"], device=device)

    def on_experiment_end(self, model: nn.Module, datamodule: DataModule, trainer: Trainer, fit_output: FitOutput,
                          experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger):
        if config['is_ugs_mask_train']:
            final_mask = ugs_utils.get_final_mask_epoch(model, config['ugs_remove_percent'], datamodule.base_num_edges)
            removed_mask = ~ final_mask.type(torch.bool)
            newly_removed_edges = datamodule.undirected_edges[:, removed_mask].tolist()

            removed_edges = [datamodule.removed_edges[0] + newly_removed_edges[0],
                             datamodule.removed_edges[1] + newly_removed_edges[1]]

            output_data = {
                'dataset': datamodule.dataset_name,
                'chunk_size': torch.sum(final_mask).item(),
                'removed_edges': removed_edges,
                'remove_undirected': False
            }
            total_removal = config['edges_ratio'] - config['ugs_remove_percent']
            with open(config['ugs_output_path'].format(edges_ratio=total_removal), 'w') as f:
                json.dump(output_data, f, indent=4)
