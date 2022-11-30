import argparse
import json
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_COLORS = ["chocolate", "g", "tab:blue", "firebrick", "tab:red"]


class ExperimentConfig:

    def __init__(self, checkpoint_path: str, metric_names: List[str], labels: List[str] = None, colors: Union[str, List[str]] = "summer",
                 linestyles: List[str] = None):
        """
        :param checkpoint_path: Path to experiment checkpoint.
        :param metric_names: Name of metrics to plot. If multiple experiments are given will take only the first metric
        (multiple metrics for multiple experiments is not supported).
        :param labels: Label for each experiment if multiple are given or label for each metric if multiple are given.
        :param colors: Color map to use for plotting multiple metrics of same run,
        or colors to use when plotting multiple metrics from different runs.
        :param linestyles: Linestyles to use. If None given will use solid for all.
        """
        self.checkpoint_path = checkpoint_path
        self.metric_names = metric_names
        self.labels = labels
        self.colors = colors
        self.linestyles = linestyles

    @staticmethod
    def load_from(experiment_config_json):
        return ExperimentConfig(**experiment_config_json)


class AxConfig:

    def __init__(self, experiment_configs: List[ExperimentConfig], start_iter: int = 0, max_iter: int = -1, min_loss: float = None,
                 loss_metric_name: str = "", title: str = "", y_label: str = "", y_top_lim: float = None, y_bottom_lim: float = None,
                 y_ticks: List[float] = None, y_ticks_labels: List[str] = None, x_label: str = "", x_ticks: List[float] = None,
                 x_ticks_labels: List[str] = None, x_tight: bool = True, plot_linewidth: float = 1.5, title_font_size: int = 12,
                 y_label_font_size: int = 11, x_label_font_size: int = 11, ticks_font_size: int = 9, legend_font_size: int = 10):
        """
        :param experiment_configs: Path to the experiments checkpoints.
        :param start_iter: Start plot from given iteration.
        :param max_iter: Maximal number of iterations to plot. If -1 will not truncate according to iterations.
        :param min_loss: Minimal loss value to plot until.
        :param loss_metric_name: Name of the loss metric.
        :param title: Title for the plot.
        :param y_label: y axis label.
        :param y_top_lim: Top limit for y axis.
        :param y_bottom_lim: Bottom limit for y axis
        :param y_ticks: y axis ticks.
        :param y_ticks_labels: labels of y axis ticks.
        :param x_label: x axis label.
        :param x_ticks: x axis ticks.
        :param x_ticks_labels: Labels of x axis ticks.
        :param x_tight: x axis edges are tight to first and last pltted values.
        :param plot_linewidth: Plots line width.
        :param title_font_size:
        :param y_label_font_size: y label font size.
        :param x_label_font_size: x label font size.
        :param ticks_font_size: ticks font size.
        :param legend_font_size: legend font size.
        """
        self.experiment_configs = experiment_configs
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.min_loss = min_loss
        self.loss_metric_name = loss_metric_name
        self.title = title
        self.y_label = y_label
        self.y_top_lim = y_top_lim
        self.y_ticks = y_ticks
        self.y_ticks_labels = y_ticks_labels
        self.y_bottom_lim = y_bottom_lim
        self.x_label = x_label
        self.x_ticks = x_ticks
        self.x_ticks_labels = x_ticks_labels
        self.x_tight = x_tight

        self.title_font_size = title_font_size
        self.y_label_font_size = y_label_font_size
        self.x_label_font_size = x_label_font_size
        self.ticks_font_size = ticks_font_size
        self.legend_font_size = legend_font_size
        self.plot_linewidth = plot_linewidth

    @staticmethod
    def load_from(ax_config_json: dict):
        experiment_configs = [ExperimentConfig.load_from(experiment_config_json) for experiment_config_json in ax_config_json["experiment_configs"]]
        ax_config_json_copy = ax_config_json.copy()
        del ax_config_json_copy["experiment_configs"]
        return AxConfig(experiment_configs, **ax_config_json_copy)


class PlotConfig:

    def __init__(self, ax_configs: List[AxConfig], title: str = "", title_font_size: int = 13, subplot_width: int = 3, subplot_height: int = 2,
                 pad_width: float = None, save_plot_to: str = ""):
        """
        :param ax_configs: Configurations for the subplots in the figure.
        :param title: Title of plot.
        :param title_font_size: Title font size.
        :param subplot_width: Width of subplots.
        :param subplot_height: Height of subplots.
        :param pad_width: Pad distance between axes.
        :param save_plot_to: Path to save plot in. If none given will not save the plot.
        """
        self.ax_configs = ax_configs
        self.title = title
        self.title_font_size = title_font_size
        self.subplot_width = subplot_width
        self.subplot_height = subplot_height
        self.pad_width = pad_width
        self.save_plot_to = save_plot_to

    @staticmethod
    def load_from(plot_config_json: dict):
        ax_configs = [AxConfig.load_from(ax_config_json) for ax_config_json in plot_config_json["ax_configs"]]
        plot_config_json_copy = plot_config_json.copy()
        del plot_config_json_copy["ax_configs"]
        return PlotConfig(ax_configs, **plot_config_json_copy)


def __set_size(w, h, ax):
    """ w, h: width, height in inches """
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def __populate_ax(ax, ax_config: AxConfig, plot_config: PlotConfig):
    per_exp_iterations_seqs, per_exp_metric_values_seqs = __extract_plot_info_from_evaluators(ax_config)

    __populate_ax_plots(ax, per_exp_iterations_seqs, per_exp_metric_values_seqs, ax_config)
    __set_ax_metadata(ax, ax_config)
    __set_size(plot_config.subplot_width, plot_config.subplot_height, ax=ax)


def __extract_plot_info_from_evaluators(ax_config: AxConfig):
    per_exp_iterations_seqs, per_exp_metric_values_seqs = [], []

    for exp_config in ax_config.experiment_configs:
        checkpoint = torch.load(exp_config.checkpoint_path, map_location=torch.device("cpu"))
        train_tracked_values = checkpoint["train_evaluator"]
        val_tracked_values = checkpoint["val_evaluator"]

        epoch_after_min_loss = __get_epoch_after_min_loss(ax_config, train_tracked_values) if ax_config.loss_metric_name else -1

        metric_epochs_seqs = []
        metric_values_seqs = []
        for metric_name in exp_config.metric_names:
            metric_tracked_value = val_tracked_values[metric_name] if metric_name in val_tracked_values else train_tracked_values[metric_name]
            metric_epochs = metric_tracked_value["epochs_with_values"]
            metric_values = metric_tracked_value["epoch_values"]

            metric_epochs, metric_values = __truncate_after_min_loss_or_max_iter(metric_epochs, metric_values, epoch_after_min_loss,
                                                                                 start_iter=ax_config.start_iter,
                                                                                 max_iter=ax_config.max_iter)
            metric_epochs_seqs.append(metric_epochs)
            metric_values_seqs.append(metric_values)

        per_exp_iterations_seqs.append(metric_epochs_seqs)
        per_exp_metric_values_seqs.append(metric_values_seqs)

    return per_exp_iterations_seqs, per_exp_metric_values_seqs


def __populate_ax_plots(ax, per_exp_iterations_seqs, per_exp_metric_values_seqs, ax_config: AxConfig):
    for i, exp_config in enumerate(ax_config.experiment_configs):
        linestyles = exp_config.linestyles if exp_config.linestyles is not None else ["solid"]
        iterations_seqs = per_exp_iterations_seqs[i]
        metric_values_seqs = per_exp_metric_values_seqs[i]

        for j, metric_values in enumerate(metric_values_seqs):
            if isinstance(exp_config.colors, str):
                color_scale = j / 8 if j < 5 else 0.5 + (j - 4) / 20
                color = plt.cm.get_cmap(exp_config.colors)(color_scale)
            elif isinstance(exp_config.colors, List):
                color = exp_config.colors[j % len(exp_config.colors)]
            else:
                color = DEFAULT_COLORS[j % len(DEFAULT_COLORS)]

            ax.plot(iterations_seqs[j], metric_values, label=exp_config.labels[j] if exp_config.labels else "", color=color,
                    linestyle=linestyles[j % len(linestyles)], linewidth=ax_config.plot_linewidth)

    if any([exp_config.labels is not None for exp_config in ax_config.experiment_configs]):
        ax.legend(prop={"size": ax_config.legend_font_size})


def __set_ax_metadata(ax, ax_config):
    ax.set_title(ax_config.title, fontsize=ax_config.title_font_size, pad=8)

    # y axis
    ax.set_ylabel(ax_config.y_label, fontsize=ax_config.y_label_font_size)
    if ax_config.y_ticks is not None:
        ax.set_yticks(ax_config.y_ticks)
    if ax_config.y_ticks_labels is not None:
        ax.set_ytickslabels(ax_config.y_ticks_labels)
    if ax_config.y_top_lim is not None:
        ax.set_ylim(top=ax_config.y_top_lim)
    if ax_config.y_bottom_lim is not None:
        ax.set_ylim(bottom=ax_config.y_bottom_lim)

    # x axis
    ax.set_xlabel(ax_config.x_label, fontsize=ax_config.x_label_font_size)
    if ax_config.x_ticks is not None:
        ax.set_xticks(ax_config.x_ticks)
    if ax_config.x_ticks_labels is not None:
        ax.set_xtickslabels(ax_config.x_ticks_labels)
    if ax_config.x_tight:
        ax.autoscale(enable=True, axis='x', tight=True)

    ax.tick_params(labelsize=ax_config.ticks_font_size)


def __get_epoch_after_min_loss(ax_config, train_tracked_values):
    loss_tracked_value = train_tracked_values[ax_config.loss_metric_name]
    loss_epochs = loss_tracked_value["epochs_with_values"]
    loss_values = loss_tracked_value["epoch_values"]

    if ax_config.min_loss is None:
        return loss_epochs[-1]

    for index, loss_value in enumerate(loss_values):
        if loss_value < ax_config.min_loss:
            return loss_epochs[index]

    return loss_epochs[-1]


def __truncate_after_min_loss_or_max_iter(metric_epochs, metric_values, epoch_after_min_loss: int = -1, start_iter: int = 0, max_iter: int = -1):
    epoch_after_min_loss = epoch_after_min_loss if epoch_after_min_loss >= 0 else metric_epochs[-1]
    np_epochs = np.array(metric_epochs)
    truncate_at = epoch_after_min_loss if max_iter < 0 else min(epoch_after_min_loss, max_iter)

    index_of_start_epoch = np_epochs.searchsorted(start_iter, side="left")
    num_before_max_iter = np_epochs.searchsorted(truncate_at, side="right")
    return metric_epochs[index_of_start_epoch: num_before_max_iter], metric_values[index_of_start_epoch: num_before_max_iter]


def create_plot_from_config(plot_config: PlotConfig):
    fig, axes = plt.subplots(1, len(plot_config.ax_configs))

    axes = axes if len(plot_config.ax_configs) > 1 else [axes]
    for ax, ax_config in zip(axes, plot_config.ax_configs):
        __populate_ax(ax, ax_config, plot_config)

    plt.suptitle(plot_config.title, fontsize=plot_config.title_font_size)
    plt.tight_layout(w_pad=plot_config.pad_width)
    if plot_config.save_plot_to:
        plt.savefig(plot_config.save_plot_to, dpi=250, bbox_inches='tight', pad_inches=0.1)

    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot_config_path", type=str, required=True, help="Path to plot config json.")
    args = p.parse_args()

    with open(args.plot_config_path) as f:
        plot_config_json = json.load(f)
        plot_config = PlotConfig.load_from(plot_config_json)
        create_plot_from_config(plot_config)


if __name__ == "__main__":
    main()
