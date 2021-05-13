import pickle
from pathlib import Path
from argparse import ArgumentParser

from survival4D.config import HypersearchConfig, NNExperimentConfig
from survival4D.nn import hypersearch_nn

DEFAULT_CONF_PATH = Path(__file__).parent.joinpath("default_nn.conf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--conf-path", dest="conf_path", type=str, default=None, help="Conf path."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.conf_path is None:
        conf_path = DEFAULT_CONF_PATH
    else:
        conf_path = Path(args.conf_path)
    exp_config = NNExperimentConfig.from_conf(conf_path)
    hypersearch_config = HypersearchConfig.from_conf(conf_path)
    with open(str(exp_config.data_path), 'rb') as f:
        c3 = pickle.load(f)
    x_full = c3[0]
    y_full = c3[1]
    del c3

    opars, clog = hypersearch_nn(
        x_data=x_full, y_data=y_full,
        method='particle swarm', nfolds=exp_config.n_folds, nevals=exp_config.n_evals,
        batch_size=exp_config.batch_size, num_epochs=exp_config.n_epochs,
        **hypersearch_config.to_dict()
    )


if __name__ == '__main__':
    main()
