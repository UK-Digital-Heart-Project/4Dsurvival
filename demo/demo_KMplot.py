import numpy as np
import pandas as pd
import pickle
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from survival4D.paths import DATA_DIR

with open(str(DATA_DIR.joinpath("bootout_conv.pkl")), 'rb') as f: inputdata_list=pickle.load(f)
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
