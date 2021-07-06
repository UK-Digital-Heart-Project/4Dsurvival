"""
@author: gbello & lisuru6
How to run the code
python demo_validateDL.py -c /path-to-conf

Default conf uses demo/scripts/default_validate_DL.conf

"""
import json
import shutil
from datetime import timedelta
import pickle
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from lifelines.utils import concordance_index

from survival4D.cox_reg import hypersearch_cox
from survival4D.cox_reg import train_cox_reg
from survival4D.config import ExperimentConfig, HypersearchConfig, ModelConfig
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

DEFAULT_CONF_PATH = Path(__file__).parent.joinpath("default_cox.conf")


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
    exp_config = ExperimentConfig.from_conf(conf_path)
    exp_config.output_dir.mkdir(parents=True, exist_ok=True)
    hypersearch_config = HypersearchConfig.from_conf(conf_path)

    shutil.copy(str(conf_path), str(exp_config.output_dir.joinpath("nn.conf")))

    # import input data: i_full=list of patient IDs, y_full=censoring status and survival times for patients,
    # x_full=input data for patients (i.e. motion descriptors [11,514-element vector])

    with open(str(exp_config.data_path), 'rb') as f:
        c3 = pickle.load(f)
    x_full = c3[0]
    y_full = c3[1]
    del c3

    # Initialize lists to store predictions
    c_vals = []
    c_trains = []

    kf = KFold(n_splits=exp_config.n_folds)
    i = 0
    for train_indices, test_indices in kf.split(x_full):
        x_train, y_train = x_full[train_indices], y_full[train_indices]
        x_val, y_val = x_full[test_indices], y_full[test_indices]

        # STEP 1: find optimal hyperparameters using CV
        print("Step 1a")
        opars, osummary = hypersearch_cox(
            x_data=x_train,
            y_data=y_train,
            method=exp_config.search_method,
            nfolds=exp_config.n_folds,
            nevals=exp_config.n_evals,
            penalty_range=hypersearch_config.penalty_exp
        )
        print("Step b")
        # (1b) using optimal hyperparameters, train a model and test its performance on the holdout validation set.
        olog = train_cox_reg(
            xtr=x_train,
            ytr=y_train,
            penalty=10 ** opars['penalty'],
        )

        # (1c) Compute Harrell's Concordance index
        pred_val = olog.predict_partial_hazard(x_val)
        c_val = concordance_index(y_val[:, 1], -pred_val, y_val[:, 0])

        pred_train = olog.predict_partial_hazard(x_train)
        c_train = concordance_index(y_train[:, 1], -pred_train, y_train[:, 0])
        c_vals.append(c_val)
        c_trains.append(c_train)
        save_params(
            opars, osummary, "cv_{}".format(i), exp_config.output_dir,
            c_val=c_val, c_train=c_train,
            c_val_mean=np.mean(c_vals), c_val_var=np.var(c_vals),
            c_train_mean=np.mean(c_trains), c_train_var=np.var(c_trains)
        )
        print('Validation concordance index = {0:.4f}'.format(c_val))
        i += 1
        plot_cs(c_trains, c_vals, exp_config.output_dir)
    print('Mean Validation concordance index = {0:.4f}'.format(np.mean(c_vals)))
    print('Variance = {0:.4f}'.format(np.var(c_vals)))


def save_params(params: dict, search_log, name: str, output_dir: Path, **kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    params["search_log_optimum_c_index"] = search_log.optimum
    params["num_evals"] = search_log.stats["num_evals"]
    params["time"] = str(timedelta(seconds=search_log.stats["time"]))
    params["call_log"] = search_log.call_log
    for key in kwargs.keys():
        params[key] = kwargs[key]
    with open(str(output_dir.joinpath(name + ".json")), "w") as fp:
        json.dump(params, fp, indent=4)


def compute_bootstrap_adjusted_c_index(C_app, Cb_opts):
    # Compute bootstrap-estimated optimism (mean of optimism estimates across the B bootstrap samples)
    C_opt = np.mean(Cb_opts)

    # Adjust apparent C using bootstrap-estimated optimism
    C_adj = C_app - C_opt

    # compute confidence intervals for optimism-adjusted C
    C_opt_95confint = np.percentile([C_app - o for o in Cb_opts], q=[2.5, 97.5])

    return C_opt, C_adj, C_opt_95confint


def plot_cs(c_trains, c_vals, output_dir):
    plt.figure()
    plt.title("CV validation, mean={:.4f}, var={:.4f}".format(np.mean(c_vals), np.var(c_vals)))
    plt.plot(range(len(c_vals)), c_vals, 'rx-')

    plt.plot(range(len(c_trains)), c_trains, 'bx-')
    plt.savefig(str(output_dir.joinpath("c_train_val.png")))


if __name__ == '__main__':
    main()
