"""
@author: gbello
"""

import pickle
from lifelines.utils import concordance_index
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from survival4D.CoxReg_Single_run import coxreg_single_run
from survival4D.hypersearch import hypersearch_cox
from survival4D.paths import DATA_DIR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", dest="data_dir", type=str, default=None, help="Directory where the data file is."
    )
    parser.add_argument(
        "-f", "--file-name", dest="file_name", type=str, default="inputdata_DL.pkl", help="Data file name."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(args.data_dir)
    # import input data: i_full=list of patient IDs, y_full=censoring status and survival times for patients,
    # x_full=input data for patients (i.e. volumetric measures of RV function)
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
    opars, osummary = hypersearch_cox(
        x_data=x_full, y_data=y_full, method='particle swarm', nfolds=6, nevals=50, penalty_range=[-2,1]
    )

    # (1b) using optimal hyperparameters, train a model on full sample
    omod = coxreg_single_run(xtr=x_full, ytr=y_full, penalty=10**opars['penalty'])

    # (1c) Compute Harrell's Concordance index
    predfull = omod.predict_partial_hazard(x_full)
    C_app = concordance_index(y_full[:,1], -predfull, y_full[:,0])

    print('\n\n==================================================')
    print('Apparent concordance index = {0:.4f}'.format(C_app))
    print('==================================================\n\n')

    # BOOTSTRAP SAMPLING

    # define useful variables
    nsmp = len(x_full)
    rowids = [_ for _ in range(nsmp)]
    B = 100

    for b in range(B):
        print('\n-------------------------------------')
        print('Current bootstrap sample:', b, 'of', B-1)
        print('-------------------------------------')

        # STEP 2: Generate a bootstrap sample by doing n random selections with replacement (where n is the sample size)
        b_inds = np.random.choice(rowids, size=nsmp, replace=True)
        xboot = x_full[b_inds]
        yboot = y_full[b_inds]

        # (2a) find optimal hyperparameters
        bpars, bsummary = hypersearch_cox(
            x_data=xboot, y_data=yboot, method='particle swarm', nfolds=6, nevals=50, penalty_range=[-2,1]
        )

        # (2b) using optimal hyperparameters, train a model on bootstrap sample
        bmod = coxreg_single_run(xtr=xboot, ytr=yboot, penalty=10**bpars['penalty'])

        # (2c[i])  Using bootstrap-trained model, compute predictions on bootstrap sample.
        # Evaluate accuracy of predictions (Harrell's Concordance index)
        predboot = bmod.predict_partial_hazard(xboot)
        Cb_boot = concordance_index(yboot[:,1], -predboot, yboot[:,0])

        # (2c[ii]) Using bootstrap-trained model, compute predictions on FULL sample.
        # Evaluate accuracy of predictions (Harrell's Concordance index)
        predbootfull = bmod.predict_partial_hazard(x_full)
        Cb_full = concordance_index(y_full[:,1], -predbootfull, y_full[:,0])

        # STEP 3: Compute optimism for bth bootstrap sample, as difference between results from 2c[i] and 2c[ii]
        Cb_opt = Cb_boot - Cb_full

        # store data on current bootstrap sample (predictions, C-indices)
        preds_bootfull.append(predbootfull)
        inds_inbag.append(b_inds)
        Cb_opts.append(Cb_opt)

        del bpars, bmod

    # STEP 5
    # Compute bootstrap-estimated optimism (mean of optimism estimates across the B bootstrap samples)
    C_opt = np.mean(Cb_opts)

    # Adjust apparent C using bootstrap-estimated optimism
    C_adj = C_app - C_opt

    # compute confidence intervals for optimism-adjusted C
    C_opt_95confint = np.percentile([C_app - o for o in Cb_opts], q=[2.5, 97.5])

    print('\n\n=========================SUMMARY=========================')
    print('Apparent concordance index = {0:.4f}'.format(C_app))
    print('Optimism bootstrap estimate = {0:.4f}'.format(C_opt))
    print('Optimism-adjusted concordance index = {0:.4f}, and 95% CI = {1}'.format(C_adj, C_opt_95confint))


if __name__ == '__main__':
    main()
