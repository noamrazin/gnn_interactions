import logging
from typing import Tuple

import torch
import torch.utils.data
from torch import nn as nn

from common.data.modules import DataModule
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import Evaluator, TrainEvaluator
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.trainer import Trainer
from evaluation.torch_geometric_supervised_evaluator import TorchGeometricSupervisedValidationEvaluator, \
    TorchGeometricSupervisedTrainEvaluator
from is_same_class.datasets.is_same_class_datamodule import IsSameClassDataModule
from is_same_class.graph_model import GraphModel
from is_same_class.utils import GNN_TYPE
from train.torch_geometric_supervised_trainer import TorchGeometricSupervisedTrainer


class IsSameClassExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset file")
        parser.add_argument("--num_train_samples", type=int, default=-1,
                            help="Number of training samples to use. If < 0 will use the whole training set")

        parser.add_argument("--model", default=GNN_TYPE.GCN, type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                            help="Which GNN model to use")
        parser.add_argument("--hidden_dim", default=16, type=int, help="Size of the hidden dimension")
        parser.add_argument("--num_layers", default=3, type=int, help="Number of layers to use (depth of network)")
        parser.add_argument("--use_layer_norm", action="store_true", help="Use layer norm if True")

        parser.add_argument("--partition_type", type=str, default="low_walk",
                            help="Determines how to distribute image patches to vertices. Supports 'low_walk' and 'high_walk'.")

        parser.add_argument("--load_dataset_to_gpu", action="store_true", help="Stores all dataset on the main GPU (if GPU device is given)")
        parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers to use")
        parser.add_argument("--batch_size", type=int, default=-1, help="Batch size. If < 0 (default) will use whole training set at each batch.")
        parser.add_argument("--accum_grad", default=1, type=int, required=False, help="Number of steps to accumulate gradients")
        parser.add_argument("--optimizer", type=str, default="adam", help="optimizer to use. Supports: 'sgd' and 'adam'.")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        parser.add_argument("--weight_decay", type=float, default=0, help="L2 regularization coefficient")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None
        datamodule = IsSameClassDataModule(dataset_path=config["dataset_path"],
                                           batch_size=config["batch_size"],
                                           partition_type=config["partition_type"],
                                           num_train_samples=config["num_train_samples"],
                                           dataloader_num_workers=config["dataloader_num_workers"],
                                           load_dataset_to_device=load_dataset_to_device)
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: IsSameClassDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        num_layers = config["num_layers"]
        gnn_type = GNN_TYPE.from_string(config["model"])
        model = GraphModel(gnn_type=gnn_type, num_layers=num_layers, dim0=datamodule.dataset.in_dim, h_dim=config["hidden_dim"],
                           out_dim=1, layer_norm=config["use_layer_norm"], use_activation=True, graph_output=True)
        return model

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: IsSameClassDataModule,
                                               val_dataloader: torch.utils.data.DataLoader, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [
            metrics.MetricInfo("train loss", metrics.BCEWithLogitsLoss(), tag="loss"),
            metrics.MetricInfo("train accuracy", metrics.BinaryClassificationAccuracyWithLogits(), tag="accuracy")
        ]

        train_evaluator = TorchGeometricSupervisedTrainEvaluator(train_metric_info_seq)

        val_metric_info_seq = [
            metrics.MetricInfo("test loss", metrics.BCEWithLogitsLoss(), tag="loss"),
            metrics.MetricInfo("test accuracy", metrics.BinaryClassificationAccuracyWithLogits(), tag="accuracy")
        ]

        val_evaluator = TorchGeometricSupervisedValidationEvaluator(model, val_dataloader, metric_info_seq=val_metric_info_seq, device=device)
        return train_evaluator, val_evaluator

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train accuracy", is_train_metric=False, largest=True, return_best_score=False)

    def create_additional_metadata_to_log(self, model: nn.Module, datamodule: IsSameClassDataModule, config: dict, state: dict,
                                          logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["input dim"] = datamodule.dataset.in_dim
        additional_metadata["train dataset size"] = len(datamodule.dataset.train_data_list)
        additional_metadata["test dataset size"] = len(datamodule.dataset.test_data_list)
        return additional_metadata

    def create_trainer(self, model: nn.Module, datamodule: IsSameClassDataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        bce_loss = nn.BCEWithLogitsLoss()
        loss_fn = lambda y_pred, y: bce_loss(y_pred.squeeze(dim=1), y)
        return TorchGeometricSupervisedTrainer(model, optimizer, loss_fn, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                               callback=callback, gradient_accumulation=config["accum_grad"], device=device)
