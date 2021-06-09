import optunity
import numpy as np
from keras import backend as K
from lifelines.utils import concordance_index
from survival4D.models import model_factory


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


def negative_log_likelihood(E, risk):
    """
    Define Cox PH partial likelihood function loss.
    Arguments: E (censoring status), risk (risk [log hazard ratio] predicted by network) for batch of input subjects
    As defined, this function requires that all subjects in input batch must be sorted in descending order of
    survival/censoring time (i.e. arguments E and risk will be in this order)
    """
    hazard_ratio = K.exp(risk)
    log_risk = K.log(K.cumsum(hazard_ratio))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * E
    neg_likelihood = -K.sum(censored_likelihood)
    return neg_likelihood


def train_nn(xtr, ytr, batch_size, n_epochs, model_name, **model_kwargs):
    """
    Data preparation: create X, E and TM where X=input vector, E=censoring status and T=survival time.
    Apply formatting (X and T as 'float32', E as 'int32')
    """
    X_tr, E_tr, TM_tr = prepare_data(xtr, ytr[:, 0, np.newaxis], ytr[:, 1])

    # Arrange data into minibatches (based on specified batch size), and within each minibatch,
    # sort in descending order of survival/censoring time (see explanation of Cox PH loss function definition)
    # X_tr, E_tr, TM_tr, _ = sort4minibatches(X_tr, E_tr, TM_tr, batch_size)

    # specify input dimensionality
    inpshape = xtr.shape[1]

    # Define Network Architecture
    model_kwargs["input_shape"] = inpshape
    model = model_factory(model_name, **model_kwargs)

    # Run model
    mlog = model.fit(X_tr, [X_tr, E_tr], batch_size=batch_size, epochs=n_epochs, shuffle=False, verbose=1)

    return mlog


# 1. Hyperparameter search for Deep Learning model
def hypersearch_nn(x_data, y_data, method, nfolds, nevals, batch_size, num_epochs, model_kwargs: dict, **hypersearch):
    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, **hypersearch):
        cv_log = train_nn(
            xtr=x_train, ytr=y_train, batch_size=batch_size, n_epochs=num_epochs, **model_kwargs, **hypersearch
        )
        cv_preds = cv_log.model.predict(x_test, batch_size=1)[1]
        cv_C = concordance_index(y_test[:, 1], -cv_preds, y_test[:, 0])
        return cv_C
    optimal_pars, searchlog, _ = optunity.maximize(
        modelrun, num_evals=nevals, solver_name=method, **hypersearch
    )
    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog
