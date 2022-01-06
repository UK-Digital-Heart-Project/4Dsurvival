import numpy as np
import torch

from survival4D.nn.torch.models import model_factory
from survival4D.nn import prepare_data, sort4minibatches
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm


def negative_log_likelihood(E, risk):
    """
    Define Cox PH partial likelihood function loss.
    Arguments: E (censoring status), risk (risk [log hazard ratio] predicted by network) for batch of input subjects
    As defined, this function requires that all subjects in input batch must be sorted in descending order of
    survival/censoring time (i.e. arguments E and risk will be in this order)
    """
    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * E
    neg_likelihood = -torch.mean(censored_likelihood)
    return neg_likelihood


def train_nn(xtr, ytr, batch_size, n_epochs, model_name, lr_exp, alpha, weight_decay_exp, **model_kwargs):
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

    dataset = TensorDataset(torch.from_numpy(X_tr).cuda(), torch.from_numpy(E_tr).cuda(), torch.from_numpy(TM_tr).cuda())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.cuda()
    # Model compilation
    optimizer = torch.optim.Adam(model.parameters(), lr=10**lr_exp, weight_decay=10**weight_decay_exp)
    loss_mse = torch.nn.MSELoss()

    for n in range(n_epochs):
        loss_ac = 0
        loss_mse_ac = 0
        loss_neg_log_ac = 0
        npoints = 0
        pbar = tqdm(enumerate(dataloader))
        for idx, (x, e, t) in pbar:
            # sort x (B, 11514), e (B, 1), t (B,) according to t
            sort_idx = torch.argsort(t, descending=True, dim=0)
            x = x[sort_idx, :]
            e = e[sort_idx, :]
            decoded, risk_pred = model(x)
            mse_loss = loss_mse(x, decoded)
            neg_log = negative_log_likelihood(e, risk_pred)
            loss = mse_loss * alpha + (1 - alpha) * neg_log

            optimizer.zero_grad()
            loss_corrected = loss
            actual_batch_size = x.shape[1]
            if actual_batch_size != batch_size:
                loss_corrected = loss_corrected * actual_batch_size / batch_size
            loss_corrected.backward()
            optimizer.step()

            loss_ac = loss.item() * actual_batch_size + npoints * loss_ac
            loss_mse_ac = mse_loss.item() * actual_batch_size + npoints * loss_mse_ac
            loss_neg_log_ac = neg_log.item() * actual_batch_size + npoints * loss_neg_log_ac
            npoints += actual_batch_size
            pbar.set_description(
                'Epoch [{epoch}/{epochs}] :: Loss {loss:.4f}, MSE loss {mse_loss:.4f}, Neg Log {neg_log:.4f}'.format(
                    epoch=n + 1,
                    epochs=n_epochs,
                    loss=loss_ac / (idx + 1),
                    mse_loss=loss_mse_ac / (idx + 1),
                    neg_log=loss_neg_log_ac / (idx + 1),
                )
            )

    return model
