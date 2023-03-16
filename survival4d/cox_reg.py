#import optunity
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def train_cox_reg(xtr, ytr, penalty):
    df_tr = pd.DataFrame(np.concatenate((ytr, xtr),axis=1))
    df_tr.columns = ['status', 'time'] + ['X'+str(i+1) for i in range(xtr.shape[1])]
    cph = CoxPHFitter(penalizer=penalty)
    cph.fit(df_tr, duration_col='time', event_col='status')
    return cph


# 2. 'Hyperparameter' search for Cox Regression model
def hypersearch_cox(x_data, y_data, method, nfolds, nevals, penalty_range):
    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, penalty):
        cvmod = train_cox_reg(xtr=x_train, ytr=y_train, penalty=10 ** penalty)
        cv_preds = cvmod.predict_partial_hazard(x_test)
        cv_C = concordance_index(y_test[:, 1], -cv_preds, y_test[:, 0])
        return cv_C
    optimal_pars, searchlog, _ = optunity.maximize(modelrun, num_evals=nevals,
                                                   solver_name=method, penalty=penalty_range)
    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog
