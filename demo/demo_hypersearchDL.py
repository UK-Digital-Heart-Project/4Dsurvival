import pickle
import optunity
from pathlib import Path
from argparse import ArgumentParser
from lifelines.utils import concordance_index

from survival4D.paths import DATA_DIR
from survival4D.trainDL import DL_single_run


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", dest="data_dir", type=str, default=None, help="Directory where the data file is."
    )
    parser.add_argument(
        "-f", "--file-name", dest="file_name", type=str, default="inputdata_DL.pkl", help="Data file name."
    )
    return parser.parse_args()


def hypersearch_DL(x_data, y_data, method, nfolds, nevals, lrexp_range, l1rexp_range, dro_range,
                   units1_range, units2_range, alpha_range, batch_size, num_epochs):

    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, lrexp, l1rexp, dro, units1, units2, alpha):
        cv_log = DL_single_run(xtr=x_train, ytr=y_train, units1=units1, units2=units2, dro=dro, lr=10**lrexp,
                               l1r=10**l1rexp, alpha=alpha, batchsize=batch_size, numepochs=num_epochs)
        cv_preds = cv_log.model.predict(x_test, batch_size=1)[1]
        cv_C = concordance_index(y_test[:,1], -cv_preds, y_test[:,0])
        return cv_C

    optimal_pars, searchlog, _ = optunity.maximize(modelrun, num_evals=nevals, solver_name=method, lrexp=lrexp_range,
                                                   l1rexp=l1rexp_range, dro=dro_range, units1=units1_range,
                                                   units2=units2_range, alpha=alpha_range)

    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog


def main():
    args = parse_args()
    if args.data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(args.data_dir)
    with open(str(data_dir.joinpath(args.file_name)), 'rb') as f:
        c3 = pickle.load(f)
    x_full = c3[0]
    y_full = c3[1]
    del c3

    opars, clog = hypersearch_DL(
        x_data=x_full, y_data=y_full,
        method='particle swarm', nfolds=6, nevals=50,
        lrexp_range=[-6.,-4.5], l1rexp_range=[-7,-4], dro_range=[.1,.9],
        units1_range=[75,250],  units2_range=[5,20],  alpha_range=[0.3, 0.7],
        batch_size=16, num_epochs=100
    )


if __name__ == '__main__':
    main()
