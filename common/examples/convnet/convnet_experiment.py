import logging
from typing import Tuple

import torch
import torch.utils.data
from torch import nn as nn

from ...data.modules import DataModule, TorchvisionDataModule
from ...evaluation import metrics as metrics
from ...evaluation.evaluators import SupervisedTrainEvaluator, SupervisedValidationEvaluator, TrainEvaluator, Evaluator
from ...experiment import FitExperimentBase
from ...experiment.fit_experiment_base import ScoreInfo
from ...models.simple_convnet import SimpleConvnet
from ...train.callbacks import Callback
from ...train.trainer import Trainer
from ...train.trainers import SupervisedTrainer
from ...utils import model as model_utils


class ConvnetExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")
        parser.add_argument("--model", type=str, default="simple_conv",
                            help="Model to use. Default ('simple_conv') is a simple convolutional network, but supports also models specified"
                                 " in common.utils.model.create_modified_model")

        parser.add_argument("--num_train_samples", type=int, default=-1, help="number of train samples to use (if < 0 will use the whole train set).")
        parser.add_argument("--batch_size", type=int, default=64, help="train batch size")
        parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers to use")
        parser.add_argument("--optimizer", type=str, default="adam", help="optimizer to use. Supports: 'sgd' and 'adam'.")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization coefficient")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        datamodule = TorchvisionDataModule(dataset_name=config["dataset"],
                                           num_train_samples=config["num_train_samples"],
                                           batch_size=config["batch_size"],
                                           num_workers=config["dataloader_num_workers"])
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        in_size = datamodule.input_dims
        num_classes = datamodule.num_classes

        if config["model"] == "simple_conv":
            return SimpleConvnet(input_size=in_size, output_size=num_classes)

        return model_utils.create_modified_model(config["model"], input_size=in_size, output_size=num_classes, pretrained=False)

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: DataModule, val_dataloader: torch.utils.data.DataLoader,
                                               device, config: dict, state: dict, logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [
            metrics.MetricInfo("train loss", metrics.CrossEntropyLoss(), tag="loss"),
            metrics.MetricInfo("train accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy"),
            metrics.MetricInfo("train top5 accuracy", metrics.TopKAccuracyWithLogits(k=5), tag="top5 accuracy")
        ]

        train_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)

        val_metric_info_seq = [
            metrics.MetricInfo("val loss", metrics.CrossEntropyLoss(), tag="loss"),
            metrics.MetricInfo("val accuracy", metrics.TopKAccuracyWithLogits(), tag="accuracy"),
            metrics.MetricInfo("val top5 accuracy", metrics.TopKAccuracyWithLogits(k=5), tag="top5 accuracy")
        ]

        val_evaluator = SupervisedValidationEvaluator(model, val_dataloader, metric_info_seq=val_metric_info_seq, device=device)
        return train_evaluator, val_evaluator

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="val accuracy", is_train_metric=False, largest=True, return_best_score=False)

    def create_trainer(self, model: nn.Module, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        loss_fn = nn.CrossEntropyLoss()
        return SupervisedTrainer(model, optimizer, loss_fn, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                 callback=callback, device=device)
