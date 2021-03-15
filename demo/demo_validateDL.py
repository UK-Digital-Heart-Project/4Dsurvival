"""
@author: gbello
"""
import json
from matplotlib import pyplot as plt
from datetime import timedelta
import pickle
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from lifelines.utils import concordance_index

from survival4D.trainDL import DL_single_run
from survival4D.hypersearch import hypersearch_DL
from survival4D.paths import DATA_DIR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", dest="data_dir", type=str, default=None, help="Directory where the data file is."
    )
    parser.add_argument(
        "-f", "--file-name", dest="file_name", type=str, default="inputdata_DL.pkl", help="Data file name."
    )
    parser.add_argument(
        "--n-evals", dest="n_evals", type=int, default=50, help="Number of evaluations per each hyperparam fold."
    )
    parser.add_argument(
        "--dropout-max", dest="dropout_max", type=float, default=0.9, help="Maximum dropout rate."
    )
    parser.add_argument(
        "--n-bootstraps", dest="n_bootstraps", type=int, default=100, help="Number of bootstrap samples."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(args.data_dir)
    # import input data: i_full=list of patient IDs, y_full=censoring status and survival times for patients,
    # x_full=input data for patients (i.e. motion descriptors [11,514-element vector])
    dropout_max = args.dropout_max
    n_evals = args.n_evals
    n_bootstraps = args.n_bootstraps
    with open(str(data_dir.joinpath(args.file_name)), 'rb') as f:
        c3 = pickle.load(f)
    x_full = c3[0]
    y_full = c3[1]
    del c3

    # Initialize lists to store predictions
    preds_bootfull = []
    inds_inbag = []
    Cb_opts = []

    # STEP 1
    # (1a) find optimal hyperparameters
    print("Step 1a")
    opars, osummary = hypersearch_DL(
        x_data=x_full, y_data=y_full, method='particle swarm', nfolds=6, nevals=n_evals, lrexp_range=[-6., -4.5],
        l1rexp_range=[-7, -4], dro_range=[.1, dropout_max], units1_range=[75, 250], units2_range=[5, 20],
        alpha_range=[0.3, 0.7], batch_size=16, num_epochs=100
    )
    # save opars
    save_params(opars, osummary, "step_1a", data_dir.joinpath("params_output"))
    print("Step b")
    # (1b) using optimal hyperparameters, train a model on full sample
    olog = DL_single_run(
        xtr=x_full, ytr=y_full, units1=opars['units1'], units2=opars['units2'], dro=opars['dro'],
        lr=10**opars['lrexp'], l1r=10**opars['l1rexp'], alpha=opars['alpha'], batchsize=16, numepochs=100
    )

    # (1c) Compute Harrell's Concordance index
    predfull = olog.model.predict(x_full, batch_size=1)[1]
    C_app = concordance_index(y_full[:, 1], -predfull, y_full[:, 0])

    print('Apparent concordance index = {0:.4f}'.format(C_app))

    # BOOTSTRAP SAMPLING

    # define useful variables
    nsmp = len(x_full)
    rowids = [_ for _ in range(nsmp)]
    B = n_bootstraps

    for b in range(B):
        print('Current bootstrap sample:', b, 'of', B-1)
        print('-------------------------------------')

        # STEP 2: Generate a bootstrap sample by doing n random selections with replacement (where n is the sample size)
        b_inds = np.random.choice(rowids, size=nsmp, replace=True)
        xboot = x_full[b_inds]
        yboot = y_full[b_inds]

        # (2a) find optimal hyperparameters
        print("Step 2a")
        bpars, bsummary = hypersearch_DL(
            x_data=xboot, y_data=yboot, method='particle swarm', nfolds=6, nevals=n_evals, lrexp_range=[-6., -4.5],
            l1rexp_range=[-7, -4], dro_range=[.1, dropout_max], units1_range=[75, 250], units2_range=[5, 20],
            alpha_range=[0.3, 0.7], batch_size=16, num_epochs=100
        )
        save_params(bpars, bsummary, "bootstrap_{}".format(b), data_dir.joinpath("params_output"))
        # (2b) using optimal hyperparameters, train a model on bootstrap sample
        blog = DL_single_run(
            xtr=xboot, ytr=yboot, units1=bpars['units1'], units2=bpars['units2'], dro=bpars['dro'],
            lr=10**bpars['lrexp'], l1r=10**bpars['l1rexp'], alpha=bpars['alpha'], batchsize=16, numepochs=100)

        # (2c[i])  Using bootstrap-trained model, compute predictions on bootstrap sample.
        # Evaluate accuracy of predictions (Harrell's Concordance index)
        predboot = blog.model.predict(xboot, batch_size=1)[1]
        Cb_boot = concordance_index(yboot[:,1], -predboot, yboot[:, 0])

        # (2c[ii]) Using bootstrap-trained model, compute predictions on FULL sample.
        # Evaluate accuracy of predictions (Harrell's Concordance index)
        predbootfull = blog.model.predict(x_full, batch_size=1)[1]
        Cb_full = concordance_index(y_full[:, 1], -predbootfull, y_full[:, 0])

        # STEP 3: Compute optimism for bth bootstrap sample, as difference between results from 2c[i] and 2c[ii]
        Cb_opt = Cb_boot - Cb_full

        # store data on current bootstrap sample (predictions, C-indices)
        preds_bootfull.append(predbootfull)
        inds_inbag.append(b_inds)
        Cb_opts.append(Cb_opt)

        del bpars, blog

    # STEP 5
    # Compute bootstrap-estimated optimism (mean of optimism estimates across the B bootstrap samples)
    C_opt = np.mean(Cb_opts)

    # Adjust apparent C using bootstrap-estimated optimism
    C_adj = C_app - C_opt

    # compute confidence intervals for optimism-adjusted C
    C_opt_95confint = np.percentile([C_app - o for o in Cb_opts], q=[2.5, 97.5])

    print('Optimism bootstrap estimate = {0:.4f}'.format(C_opt))
    print('Optimism-adjusted concordance index = {0:.4f}, and 95% CI = {1}'.format(C_adj, C_opt_95confint))


def save_params(params: dict, search_log, name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    params["c_index"] = search_log.optimum
    params["num_evals"] = search_log.stats["num_evals"]
    params["time"] = str(timedelta(seconds=search_log.stats["time"]))
    with open(str(output_dir.joinpath(name + ".json")), "w") as fp:
        json.dump(params, fp, indent=4)


if __name__ == '__main__':
    main()
