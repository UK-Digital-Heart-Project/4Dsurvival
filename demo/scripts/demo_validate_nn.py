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

from survival4D.nn import train_nn, hypersearch_nn
from survival4D.config import NNExperimentConfig, HypersearchConfig, ModelConfig
from matplotlib import pyplot as plt


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
    exp_config.output_dir.mkdir(parents=True, exist_ok=True)
    hypersearch_config = HypersearchConfig.from_conf(conf_path)
    model_config = ModelConfig.from_conf(conf_path)

    shutil.copy(conf_path, exp_config.output_dir.joinpath("nn.conf"))

    # import input data: i_full=list of patient IDs, y_full=censoring status and survival times for patients,
    # x_full=input data for patients (i.e. motion descriptors [11,514-element vector])

    with open(str(exp_config.data_path), 'rb') as f:
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
    opars, osummary = hypersearch_nn(
        x_data=x_full,
        y_data=y_full,
        method=exp_config.search_method,
        nfolds=exp_config.n_folds,
        nevals=exp_config.n_evals,
        batch_size=exp_config.batch_size,
        num_epochs=exp_config.n_epochs,
        model_kwargs=model_config.to_dict(),
        **hypersearch_config.to_dict(),
    )
    # save opars
    print("Step b")
    # (1b) using optimal hyperparameters, train a model on full sample
    olog = train_nn(
        xtr=x_full,
        ytr=y_full,
        batch_size=exp_config.batch_size,
        n_epochs=exp_config.n_epochs,
        **model_config.to_dict(),
        **opars,
    )

    # (1c) Compute Harrell's Concordance index
    predfull = olog.model.predict(x_full, batch_size=1)[1]
    C_app = concordance_index(y_full[:, 1], -predfull, y_full[:, 0])
    save_params(opars, osummary, "step_1a", exp_config.output_dir, c_app=C_app)
    print('Apparent concordance index = {0:.4f}'.format(C_app))

    # BOOTSTRAP SAMPLING

    # define useful variables
    nsmp = len(x_full)
    rowids = [_ for _ in range(nsmp)]
    B = exp_config.n_bootstraps
    plot_c_opts = []
    plot_c_adjs = []
    plot_bs_samples = []
    plot_c_adjs_lb = []
    plot_c_adjs_up = []
    for b in range(B):
        print('Current bootstrap sample:', b, 'of', B-1)
        print('-------------------------------------')

        # STEP 2: Generate a bootstrap sample by doing n random selections with replacement (where n is the sample size)
        b_inds = np.random.choice(rowids, size=nsmp, replace=True)
        xboot = x_full[b_inds]
        yboot = y_full[b_inds]

        # (2a) find optimal hyperparameters
        print("Step 2a")
        bpars, bsummary = hypersearch_nn(
            x_data=xboot,
            y_data=xboot,
            method=exp_config.search_method,
            nfolds=exp_config.n_folds,
            nevals=exp_config.n_evals,
            batch_size=exp_config.batch_size,
            num_epochs=exp_config.n_epochs,
            model_kwargs=model_config.to_dict(),
            **hypersearch_config.to_dict(),
        )
        # (2b) using optimal hyperparameters, train a model on bootstrap sample
        blog = train_nn(
            xtr=xboot,
            ytr=xboot,
            batch_size=exp_config.batch_size,
            n_epochs=exp_config.n_epochs,
            model_name=exp_config.model_name,
            **model_config.to_dict(),
            **bpars
        )

        # (2c[i])  Using bootstrap-trained model, compute predictions on bootstrap sample.
        # Evaluate accuracy of predictions (Harrell's Concordance index)
        predboot = blog.model.predict(xboot, batch_size=1)[1]
        Cb_boot = concordance_index(yboot[:, 1], -predboot, yboot[:, 0])

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
        print('Current bootstrap sample:', b, 'of', B-1)
        print('-------------------------------------')
        c_opt, c_adj, c_opt_95confint = compute_bootstrap_adjusted_c_index(C_app, Cb_opts)
        print('Optimism bootstrap estimate = {0:.4f}'.format(c_opt))
        print('Optimism-adjusted concordance index = {0:.4f}, and 95% CI = {1}'.format(c_adj, c_opt_95confint))
        save_params(
            bpars, bsummary, "bootstrap_{}".format(b), exp_config.output_dir,
            c_opt=c_opt, c_adj=c_adj, c_opt_95confint=c_opt_95confint.tolist(),
            cb_boot=Cb_boot, cb_full=Cb_full, cb_opt=Cb_opt, c_app=C_app,
        )

        # plot c_opt, c_adj with c_app as title
        plot_c_opts.append(c_opt)
        plot_bs_samples.append(b)
        plot_c_adjs.append(c_adj)
        plot_c_adjs_lb.append(c_opt_95confint[0])
        plot_c_adjs_up.append(c_opt_95confint[1])

        plot_c_indices(plot_bs_samples, plot_c_opts, plot_c_adjs, plot_c_adjs_lb, plot_c_adjs_up, C_app, exp_config.output_dir)
        del bpars, blog

    # STEP 5
    # Compute bootstrap-estimated optimism (mean of optimism estimates across the B bootstrap samples)
    c_opt, c_adj, c_opt_95confint = compute_bootstrap_adjusted_c_index(C_app, Cb_opts)
    print('Optimism bootstrap estimate = {0:.4f}'.format(c_opt))
    print('Optimism-adjusted concordance index = {0:.4f}, and 95% CI = {1}'.format(c_adj, c_opt_95confint))


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


def plot_c_indices(bs_samples, c_obts, c_adjs, c_adjs_lb, c_adjst_up, c_app, output_dir: Path):
    plt.figure()
    plt.title("c_adj, c_app={:.4f}".format(c_app))
    plt.fill_between(bs_samples, c_adjs_lb, c_adjst_up, facecolor='red', alpha=0.5, interpolate=True)
    plt.plot(bs_samples, c_adjs, 'rx-')
    plt.savefig(str(output_dir.joinpath("c_adj.png")))

    plt.figure()
    plt.title("c_opt, c_app={:.4f}".format(c_app))
    plt.plot(bs_samples, c_obts, 'rx-')
    plt.savefig(str(output_dir.joinpath("c_obt.png")))


if __name__ == '__main__':
    main()
