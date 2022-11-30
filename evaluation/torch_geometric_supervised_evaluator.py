from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.evaluation import metrics as metrics
from common.evaluation.evaluators.evaluator import MetricsEvaluator, Evaluator, TrainEvaluator


class TorchGeometricSupervisedTrainEvaluator(TrainEvaluator):

    def __init__(self, metric_info_seq: Sequence[metrics.MetricInfo] = None):
        self.metric_infos = metrics.metric_info_seq_to_dict(metric_info_seq) if metric_info_seq is not None else {}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate_batch(self, output):
        y_pred = output["y_pred"]
        y = output["y"]

        metric_values = {}
        for name, metric in self.metrics.items():
            value = metric(y_pred, y)
            self.tracked_values[name].add_batch_value(value)
            metric_values[name] = value

        return metric_values


class TorchGeometricSupervisedValidationEvaluator(Evaluator):

    def __init__(self, model: nn.Module, data_loader: DataLoader, metric_info_seq: Sequence[metrics.MetricInfo] = None,
                 device=torch.device("cpu")):
        self.metric_infos = metrics.metric_info_seq_to_dict(metric_info_seq) if metric_info_seq is not None else {}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

        self.model = model
        self.data_loader = data_loader
        self.device = device

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate(self):
        with torch.no_grad():
            self.model.to(self.device)
            for batch in self.data_loader:
                batch = batch.to(self.device)
                y = batch.y
                y_pred = self.model(batch)

                for name, metric in self.metrics.items():
                    value = metric(y_pred, y)
                    self.tracked_values[name].add_batch_value(value)

            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
