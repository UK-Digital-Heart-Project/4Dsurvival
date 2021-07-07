import optunity
import numpy as np
from lifelines.utils import concordance_index


def prepare_data(x, e, t):
    return x.astype("float32"), e.astype("int32"), t.astype("float32")


def sort4minibatches(xvals, evals, tvals, batchsize):
    ntot = len(xvals)
    indices = np.arange(ntot)
    np.random.shuffle(indices)
    start_idx=0
    esall = []
    for end_idx in list(range(batchsize, batchsize*(ntot//batchsize)+1, batchsize))+[ntot]:
        excerpt = indices[start_idx:end_idx]
        sort_idx = np.argsort(tvals[excerpt])[::-1]
        es = excerpt[sort_idx]
        esall += list(es)
        start_idx = end_idx
    return xvals[esall], evals[esall], tvals[esall], esall


# 1. Hyperparameter search for Deep Learning model
def hypersearch_nn(x_data, y_data, method, nfolds, nevals, batch_size, num_epochs, backend: str,
                   model_kwargs: dict, **hypersearch):

    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, **hypersearch):
        cv_log = train_nn(
            backend=backend, xtr=x_train, ytr=y_train, batch_size=batch_size, n_epochs=num_epochs,
            **model_kwargs, **hypersearch
        )
        cv_preds = cv_log.predict(x_test, batch_size=1)[1]
        cv_C = concordance_index(y_test[:, 1], -cv_preds, y_test[:, 0])
        return cv_C
    optimal_pars, searchlog, _ = optunity.maximize(
        modelrun, num_evals=nevals, solver_name=method, **hypersearch
    )
    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog


def train_nn(backend: str, xtr, ytr, batch_size, n_epochs, model_name, lr_exp, alpha, weight_decay_exp, **model_kwargs):
    if backend == "tf":
        from survival4D.nn.tf import train_nn
    elif backend == "torch":
        from survival4D.nn.torch import train_nn
    else:
        raise ValueError("Backend {} not supported. Only tf or torch. ".format(backend))
    return train_nn(xtr, ytr, batch_size, n_epochs, model_name, lr_exp, alpha, weight_decay_exp, **model_kwargs)
