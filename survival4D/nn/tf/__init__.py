import numpy as np

from survival4D.nn.tf.models import model_factory
from survival4D.nn import prepare_data, sort4minibatches
import keras
from keras import backend as K
from keras.optimizers import Adam


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
    neg_likelihood = -K.mean(censored_likelihood)
    return neg_likelihood


def compile_model(model, lr_exp, alpha):
    model.summary()
    # Model compilation
    optimizer = Adam(lr=10**lr_exp)
    model.compile(
        loss=[keras.losses.mean_squared_error, negative_log_likelihood],
        loss_weights=[alpha, 1 - alpha],
        optimizer=optimizer,
        metrics={'decoded': keras.metrics.mean_squared_error}
    )
    return model


def train_nn(xtr, ytr, batch_size, n_epochs, model_name, lr_exp, alpha, **model_kwargs):
    """
    Data preparation: create X, E and TM where X=input vector, E=censoring status and T=survival time.
    Apply formatting (X and T as 'float32', E as 'int32')
    """
    X_tr, E_tr, TM_tr = prepare_data(xtr, ytr[:, 0, np.newaxis], ytr[:, 1])

    # Arrange data into minibatches (based on specified batch size), and within each minibatch,
    # sort in descending order of survival/censoring time (see explanation of Cox PH loss function definition)
    X_tr, E_tr, TM_tr, _ = sort4minibatches(X_tr, E_tr, TM_tr, batch_size)

    # specify input dimensionality
    inpshape = xtr.shape[1]

    # Define Network Architecture
    model_kwargs["input_shape"] = inpshape
    model = model_factory(model_name, **model_kwargs)
    model = compile_model(model, lr_exp, alpha)
    # Run model
    mlog = model.fit(X_tr, [X_tr, E_tr], batch_size=batch_size, epochs=n_epochs, shuffle=False, verbose=1)

    return mlog.model
