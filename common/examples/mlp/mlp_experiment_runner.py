import argparse

import common.utils.logging as logging_utils
from common.examples.mlp.mlp_experiment import MultiLayerPerceptronExperiment


def main():
    parser = argparse.ArgumentParser()
    MultiLayerPerceptronExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = MultiLayerPerceptronExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
