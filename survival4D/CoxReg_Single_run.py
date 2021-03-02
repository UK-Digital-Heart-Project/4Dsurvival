import numpy as np, pandas as pd
from lifelines import CoxPHFitter


def coxreg_single_run(xtr, ytr, penalty):
    df_tr = pd.DataFrame(np.concatenate((ytr, xtr),axis=1))
    df_tr.columns = ['status', 'time'] + ['X'+str(i+1) for i in range(xtr.shape[1])]
    cph = CoxPHFitter(penalizer=penalty)
    cph.fit(df_tr, duration_col='time', event_col='status')
    return cph
