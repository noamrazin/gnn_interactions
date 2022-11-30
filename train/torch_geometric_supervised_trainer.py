import torch

from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer


class TorchGeometricSupervisedTrainer(Trainer):

    def __init__(self, model, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 reg_fn = None, gradient_accumulation: int = -1, device=torch.device("cpu")):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn
        self.reg_fn = reg_fn
        self.gradient_accumulation = gradient_accumulation

    def batch_update(self, batch_num, batch, total_num_batches):
        batch = batch.to(self.device)
        y_pred = self.model(batch)
        y = batch.y

        loss = self.loss_fn(y_pred, y)

        if self.reg_fn:
            loss += self.reg_fn(self.model)

        if self.gradient_accumulation > 0:
            loss = loss / self.gradient_accumulation

        loss.backward()

        do_accumulated_grad_update = (batch_num + 1) % self.gradient_accumulation == 0 or batch_num == total_num_batches - 1
        if self.gradient_accumulation <= 0 or do_accumulated_grad_update:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y
        }
