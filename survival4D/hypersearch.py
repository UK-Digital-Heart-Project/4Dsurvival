import optunity
from lifelines.utils import concordance_index

from survival4D.trainDL import DL_single_run
from survival4D.CoxReg_Single_run import coxreg_single_run


# 1. Hyperparameter search for Deep Learning model
def hypersearch_DL(x_data, y_data, method, nfolds, nevals, lrexp_range, l1rexp_range, dro_range, units1_range,
                   units2_range, alpha_range, batch_size, num_epochs):
    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, lrexp, l1rexp, dro, units1, units2, alpha):
        cv_log = DL_single_run(xtr=x_train, ytr=y_train, units1=units1, units2=units2, dro=dro, lr=10**lrexp,
                               l1r=10**l1rexp, alpha=alpha, batchsize=batch_size, numepochs=num_epochs)
        cv_preds = cv_log.model.predict(x_test, batch_size=1)[1]
        cv_C = concordance_index(y_test[:, 1], -cv_preds, y_test[:, 0])
        return cv_C
    optimal_pars, searchlog, _ = optunity.maximize(
        modelrun, num_evals=nevals, solver_name=method, lrexp=lrexp_range,
        l1rexp=l1rexp_range, dro=dro_range, units1=units1_range, units2=units2_range, alpha=alpha_range
    )
    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog


# 2. 'Hyperparameter' search for Cox Regression model
def hypersearch_cox(x_data, y_data, method, nfolds, nevals, penalty_range):
    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, penalty):
        cvmod = coxreg_single_run(xtr=x_train, ytr=y_train, penalty=10**penalty)
        cv_preds = cvmod.predict_partial_hazard(x_test)
        cv_C = concordance_index(y_test[:, 1], -cv_preds, y_test[:, 0])
        return cv_C
    optimal_pars, searchlog, _ = optunity.maximize(modelrun, num_evals=nevals,
                                                   solver_name=method, penalty=penalty_range)
    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog
