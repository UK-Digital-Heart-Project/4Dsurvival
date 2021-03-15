import scipy
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from argparse import ArgumentParser
from survival4D.paths import DATA_DIR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", dest="data_dir", type=str, default=None, help="Directory where the data file is."
    )
    return parser.parse_args()


def p_reader(pfile):
    with open(pfile, 'rb') as f:
        mlist = pickle.load(f)
    return mlist[0], mlist[1]


def main():
    args = parse_args()
    if args.data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(args.data_dir)
    C_app_model1, opts_model1 = p_reader(str(data_dir.joinpath("modelCstats_DL.pkl")))
    C_app_model2, opts_model2 = p_reader(str(data_dir.joinpath("modelCstats_conv.pkl")))

    Cb_adjs_model1 = [C_app_model1 - o for o in opts_model1]
    Cb_adjs_model2 = [C_app_model2 - o for o in opts_model2]

    pval = scipy.stats.wilcoxon(Cb_adjs_model1, Cb_adjs_model2)

    print(
        'Model 1 optimism-adjusted concordance index = {0:.4f}\nModel 2 optimism-adjusted concordance index = {1:.4f}\n'
        'p-value = {2}'.format(np.mean(Cb_adjs_model1), np.mean(Cb_adjs_model2), pval.pvalue)
    )


if __name__ == '__main__':
    main()
