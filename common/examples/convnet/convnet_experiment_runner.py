import argparse

import common.utils.logging as logging_utils
from common.examples.convnet.convnet_experiment import ConvnetExperiment


def main():
    parser = argparse.ArgumentParser()
    ConvnetExperiment.add_experiment_specific_args(parser)
    args = parser.parse_args()

    experiment = ConvnetExperiment()
    experiment.run(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
