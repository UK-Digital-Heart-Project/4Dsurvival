import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1
import numpy as np


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
    return (xvals[esall], evals[esall], tvals[esall], esall)


def _negative_log_likelihood(E, risk):
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


def DL_single_run(xtr, ytr, units1, units2, dro, lr, l1r, alpha, batchsize, numepochs):
    """
    Data preparation: create X, E and TM where X=input vector, E=censoring status and T=survival time.
    Apply formatting (X and T as 'float32', E as 'int32')


    """
    X_tr, E_tr, TM_tr = prepare_data(xtr, ytr[:,0,np.newaxis], ytr[:,1])

    # Arrange data into minibatches (based on specified batch size), and within each minibatch,
    # sort in descending order of survival/censoring time (see explanation of Cox PH loss function definition)
    X_tr, E_tr, TM_tr, _ = sort4minibatches(X_tr, E_tr, TM_tr, batchsize)
    
    # Before defining network architecture, clear current computation graph (if one exists),
    # and specify input dimensionality
    K.clear_session()
    inpshape = xtr.shape[1]
    
    # Define Network Architecture
    inputvec = Input(shape=(inpshape,))
    x = Dropout(dro, input_shape=(inpshape,))(inputvec)
    x = Dense(units=int(units1), activation='relu', activity_regularizer=l1(l1r))(x)
    encoded = Dense(units=int(units2), activation='relu', name='encoded')(x)
    riskpred= Dense(units=1,  activation='linear', name='predicted_risk')(encoded)
    z = Dense(units=int(units1),  activation='relu')(encoded)
    decoded = Dense(units=inpshape, activation='linear', name='decoded')(z)

    model = Model(inputs=inputvec, outputs=[decoded,riskpred])
    model.summary()
    
    # Model compilation
    optimdef = Adam(lr=lr)
    model.compile(
        loss=[keras.losses.mean_squared_error, _negative_log_likelihood],
        loss_weights=[alpha, 1-alpha],
        optimizer=optimdef,
        metrics={'decoded':keras.metrics.mean_squared_error}
    )
    
    # Run model
    mlog = model.fit(X_tr, [X_tr,E_tr], batch_size=batchsize, epochs=numepochs, shuffle=False, verbose=1)

    return mlog
