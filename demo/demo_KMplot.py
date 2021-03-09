import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from lifelines import KaplanMeierFitter
from survival4D.paths import DATA_DIR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", dest="data_dir", type=str, default=None, help="Directory where the data file is."
    )
    parser.add_argument(
        "-f", "--file-name", dest="file_name", type=str, default="bootout_conv.pkl", help="Data file name."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(args.data_dir)
    with open(str(data_dir.joinpath(args.file_name)), 'rb') as f:
        inputdata_list = pickle.load(f)
    y_orig = inputdata_list[0]
    preds_bootfull = inputdata_list[1]
    inds_inbag = inputdata_list[2]
    del inputdata_list

    preds_bootfull_mat = np.concatenate(preds_bootfull, axis=1)
    inds_inbag_mat = np.array(inds_inbag).T
    inbag_mask = 1*np.array([np.any(inds_inbag_mat==_, axis=0) for _ in range(inds_inbag_mat.shape[0])])
    preds_bootave_oob = np.divide(np.sum(np.multiply((1-inbag_mask), preds_bootfull_mat), axis=1), np.sum(1-inbag_mask, axis=1))
    risk_groups = 1*(preds_bootave_oob > np.median(preds_bootave_oob))

    wdf = pd.DataFrame(
        np.concatenate((y_orig, preds_bootave_oob[:, np.newaxis],risk_groups[:, np.newaxis]), axis=-1),
        columns=['status', 'time', 'preds', 'risk_groups'], index=[str(_) for _ in risk_groups]
    )

    kmf = KaplanMeierFitter()
    ax = plt.subplot(111)
    kmf.fit(durations=wdf.loc['0','time'], event_observed=wdf.loc['0','status'], label="Low Risk")
    ax = kmf.plot(ax=ax)
    kmf.fit(durations=wdf.loc['1','time'], event_observed=wdf.loc['1','status'], label="High Risk")
    ax = kmf.plot(ax=ax)
    plt.ylim(0,1)
    plt.title("Kaplan-Meier Plots")
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')


if __name__ == '__main__':
    main()
