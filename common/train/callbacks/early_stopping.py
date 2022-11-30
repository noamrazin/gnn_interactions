from __future__ import annotations

import copy
import operator
from typing import Callable, TYPE_CHECKING

import numpy as np

from .callback import Callback
from ..stop_fit_iteration import StopFitIteration

if TYPE_CHECKING:
    from ..trainer import Trainer


class EarlyStopping(Callback):
    """
    Will stop training when a monitored quantity has stopped improving.
    """

    def __init__(self, score_func: Callable[[Trainer], float], score_name: str = "", largest: bool = True, min_delta: float = 0, patience: int = 0,
                 cooldown: int = 0, validate_every: int = 1, restore_best_weights: bool = False):
        """
        :param score_func: callable that takes a trainer as a parameter and returns a score for it.
        :param score_name: name of the score metric (used for StopFitIteration message).
        :param largest: flag whether largest score value is better, false for smallest.
        :param min_delta: minimum change to be considered an improvement in an epoch.
        :param patience: number of checks with no improvement after which training will be stopped.
        :param cooldown: number of epochs at beginning of training to not check for improvement.
        :param validate_every: epoch interval to validate early stopping condition every this number of epochs.
        :param restore_best_weights: flag whether to restore model weights from the epoch with the best score value. If False, the model weights
        obtained at the last step of training are used.
        """
        self.score_func = score_func
        self.score_name = score_name
        self.largest = largest
        self.patience = patience
        self.cooldown = cooldown
        self.validate_every = validate_every
        self.restore_best_weights = restore_best_weights

        self.best_model_state = None
        self.num_not_improved_in_a_row = 0
        self.min_delta = min_delta if self.largest else -min_delta
        self.best_score = -np.inf if self.largest else np.inf
        self.score_is_better_op = operator.gt if self.largest else operator.lt

    def on_fit_start(self, trainer, num_epochs):
        if self.restore_best_weights:
            self.best_model_state = copy.deepcopy(trainer.model.state_dict())

    def on_fit_end(self, trainer, num_epochs_ran, fit_output):
        if self.restore_best_weights and self.best_model_state is not None:
            trainer.model.load_state_dict(self.best_model_state)

    def on_epoch_end(self, trainer):
        if trainer.epoch < self.cooldown:
            return

        if (trainer.epoch + 1) % self.validate_every == 0:
            self.__early_stopping_check(trainer)

    def __early_stopping_check(self, trainer):
        cur_score = self.score_func(trainer)
        if self.score_is_better_op(cur_score - self.min_delta, self.best_score):
            self.num_not_improved_in_a_row = 0
            self.best_score = cur_score

            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(trainer.model.state_dict())
        else:
            self.num_not_improved_in_a_row += 1

        if self.num_not_improved_in_a_row > self.patience:
            self.__early_stop(trainer.epoch)

    def __early_stop(self, epoch):
        score_name_str = self.score_name if self.score_name else "score"
        raise StopFitIteration(f"Early stopping at end of epoch {epoch} because {score_name_str} has not improved in "
                               f"{self.num_not_improved_in_a_row} validations in a row")
