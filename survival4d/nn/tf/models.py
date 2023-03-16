import keras
from typing import Tuple
from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1


def baseline_autoencoder(
    input_shape: Tuple, dropout: float, num_ae_units1: int, num_ae_units2: int, l1_reg_lambda_exp: float,
) -> Model:
    """Baseline autoencoder as published in https://www.nature.com/articles/s42256-019-0019-2"""
    inputvec = Input(shape=(input_shape,))
    x = Dropout(dropout, input_shape=(input_shape,))(inputvec)
    x = Dense(units=int(num_ae_units1), activation='relu', activity_regularizer=l1(10**l1_reg_lambda_exp))(x)
    encoded = Dense(units=int(num_ae_units2), activation='relu', name='encoded')(x)
    risk_pred = Dense(units=1,  activation='linear', name='predicted_risk')(encoded)
    z = Dense(units=int(num_ae_units1), activation='relu')(encoded)
    decoded = Dense(units=input_shape, activation='linear', name='decoded')(z)

    model = Model(inputs=inputvec, outputs=[decoded, risk_pred])
    return model


def baseline_bn_autoencoder(
    input_shape: Tuple, dropout: float, num_ae_units1: int, num_ae_units2: int, l1_reg_lambda_exp: float,
) -> Model:
    """Add batch normalization to each layer before relu activation, based on baseline_autoencoder."""
    inputvec = Input(shape=(input_shape,))
    x = Dropout(dropout, input_shape=(input_shape,))(inputvec)

    x = Dense(units=int(num_ae_units1), activation=None, activity_regularizer=l1(10**l1_reg_lambda_exp))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(units=int(num_ae_units2), activation=None)(x)
    x = BatchNormalization()(x)
    encoded = Activation("relu", name='encoded')(x)

    risk_pred = Dense(units=1,  activation='linear', name='predicted_risk')(encoded)

    x = Dense(units=int(num_ae_units1), activation=None)(encoded)
    x = BatchNormalization()(x)
    z = Activation("relu")(x)

    decoded = Dense(units=input_shape, activation='linear', name='decoded')(z)

    model = Model(inputs=inputvec, outputs=[decoded, risk_pred])
    return model


def model3_bn_autoencoder(
    input_shape: Tuple, dropout: float, num_ae_units1: int, num_ae_units2: int, num_risk_units: int,
    l1_reg_lambda_exp: float,
) -> Model:
    """
    Add one more relu layer between encoded and risk_pred, based on baseline_bn_autoencoder.
    Model 3 architecture: https://arxiv.org/pdf/1910.02951v1.pdf
    """
    inputvec = Input(shape=(input_shape,))
    x = Dropout(dropout, input_shape=(input_shape,))(inputvec)

    x = Dense(units=int(num_ae_units1), activation=None, activity_regularizer=l1(10**l1_reg_lambda_exp))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(units=int(num_ae_units2), activation=None)(x)
    x = BatchNormalization()(x)
    encoded = Activation("relu", name='encoded')(x)

    x = Dense(units=num_risk_units,  activation=None)(encoded)
    x = BatchNormalization()(x)
    x = Activation("relu", name='encoded')(x)

    risk_pred = Dense(units=1,  activation='linear', name='predicted_risk')(x)

    x = Dense(units=int(num_ae_units1), activation=None)(encoded)
    x = BatchNormalization()(x)
    z = Activation("relu")(x)

    decoded = Dense(units=input_shape, activation='linear', name='decoded')(z)

    model = Model(inputs=inputvec, outputs=[decoded, risk_pred])
    return model


def deep_model3_bn_autoencoder(
    input_shape: Tuple, dropout: float, num_ae_units1: int, num_ae_units2: int, num_ae_units3: int,
    num_risk_units: int, l1_reg_lambda_exp: float,
) -> Model:
    """
    Add one more relu layer in autoencoder, based on model3_bn_autoencoder.
    Model 3 architecture: https://arxiv.org/pdf/1910.02951v1.pdf
    """
    inputvec = Input(shape=(input_shape,))
    x = Dropout(dropout, input_shape=(input_shape,))(inputvec)

    x = Dense(units=int(num_ae_units1), activation=None, activity_regularizer=l1(10**l1_reg_lambda_exp))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(units=int(num_ae_units2), activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(units=int(num_ae_units3), activation=None)(x)
    x = BatchNormalization()(x)
    encoded = Activation("relu", name='encoded')(x)

    x = Dense(units=num_risk_units,  activation=None)(encoded)
    x = BatchNormalization()(x)
    x = Activation("relu", name='encoded')(x)

    risk_pred = Dense(units=1,  activation='linear', name='predicted_risk')(x)

    x = Dense(units=int(num_ae_units2), activation=None)(encoded)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(units=int(num_ae_units1), activation=None)(x)
    x = BatchNormalization()(x)
    z = Activation("relu")(x)

    decoded = Dense(units=input_shape, activation='linear', name='decoded')(z)

    model = Model(inputs=inputvec, outputs=[decoded, risk_pred])
    return model


def model_factory(model_name: str, **kwargs):
    # Before defining network architecture, clear current computation graph (if one exists)
    K.clear_session()
    if model_name == "baseline_autoencoder":
        model = baseline_autoencoder(**kwargs)
    elif model_name == "baseline_bn_autoencoder":
        model = baseline_bn_autoencoder(**kwargs)
    elif model_name == "model3_bn_autoencoder":
        model = model3_bn_autoencoder(**kwargs)
    elif model_name == "deep_model3_bn_autoencoder":
        model = deep_model3_bn_autoencoder(**kwargs)
    else:
        raise ValueError("Model name {} has not been implemented.".format(model_name))
    return model
