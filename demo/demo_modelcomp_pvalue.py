import scipy
from scipy.stats import wilcoxon
import numpy as np
import pickle

def p_reader(pfile):
    with open(pfile, 'rb') as f: mlist = pickle.load(f)
    return mlist[0], mlist[1]

C_app_model1, opts_model1 = p_reader('../data/modelCstats_DL.pkl')
C_app_model2, opts_model2 = p_reader('../data/modelCstats_conv.pkl')

Cb_adjs_model1 = [C_app_model1 - o for o in opts_model1]
Cb_adjs_model2 = [C_app_model2 - o for o in opts_model2]

pval = scipy.stats.wilcoxon(Cb_adjs_model1, Cb_adjs_model2)

print('Model 1 optimism-adjusted concordance index = {0:.4f}\nModel 2 optimism-adjusted concordance index = {1:.4f}\np-value = {2}'.format(np.mean(Cb_adjs_model1), np.mean(Cb_adjs_model2), pval.pvalue))
