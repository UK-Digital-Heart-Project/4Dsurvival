import numpy as np
import torch
import os

from survival4d.nn.torch.models import model_factory
from survival4d.nn import prepare_data, sort4minibatches
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
from lifelines.utils import concordance_index
from tqdm import tqdm

from copy import deepcopy

cuda_dev_id = np.random.default_rng(os.getpid()).random()
cuda_dev_id *= torch.cuda.device_count()-1
cuda_dev_id = int(np.round(cuda_dev_id))

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
    neg_likelihood = -torch.sum(censored_likelihood)
    return neg_likelihood


def train_nn_torch(xtr, ytr, batch_size, n_epochs, model_name, lr_exp, alpha, weight_decay_exp, xtr_cp=None,
    xtr_eval=None, ytr_eval=None, xtr_cp_eval=None, es_patience = 30,
    verbose=True, **model_kwargs):
    """
    Data preparation: create X, E and TM where X=input vector, E=censoring status and T=survival time.
    Apply formatting (X and T as 'float32', E as 'int32')
    """
    assert (model_name == 'baseline_bn_autoencoder_with_cp') ^ (xtr_cp is None)

    # specify input dimensionality
    inpshape = xtr.shape[1]

    # Define Network Architecture
    model_kwargs["input_shape"] = inpshape
    model = model_factory(model_name, **model_kwargs)

    device = torch.device(f"cuda:{cuda_dev_id}" if torch.cuda.is_available() else "cpu")

    X_tr, E_tr, TM_tr = prepare_data(xtr, ytr[:, 0, np.newaxis], ytr[:, 1])

    # Arrange data into minibatches (based on specified batch size), and within each minibatch,
    # sort in descending order of survival/censoring time (see explanation of Cox PH loss function definition)
    sort_idx = torch.argsort(torch.as_tensor(TM_tr), descending=True, dim=0)
    TM_tr = TM_tr[sort_idx]
    X_tr = X_tr[sort_idx, :]
    E_tr = E_tr[sort_idx, :]
    if xtr_cp is not None:
        xtr_cp = xtr_cp.astype("float32")
        xtr_cp = xtr_cp[sort_idx, :]

    dataset = [torch.from_numpy(X_tr).to(device), torch.from_numpy(E_tr).to(device), torch.from_numpy(TM_tr).to(device)]
    if xtr_cp is not None:
        dataset.append(torch.from_numpy(xtr_cp).to(device))
    dataset = TensorDataset(*dataset)

    # early stopping configuration
    if xtr_eval is not None:
        model.best_cindex_val = -np.inf
        model.loss_history_validation = []
        es_tries = 0
        X_tr, E_tr, TM_tr = prepare_data(xtr_eval, ytr_eval[:, 0, np.newaxis], ytr_eval[:, 1])
        sort_idx = torch.argsort(torch.as_tensor(TM_tr), descending=True, dim=0)
        TM_tr = TM_tr[sort_idx]
        X_tr = X_tr[sort_idx, :]
        E_tr = E_tr[sort_idx, :]

        x_eval = X_tr
        e_eval = E_tr
        t_eval = TM_tr

        if xtr_cp_eval is not None:
            xtr_cp_eval = xtr_cp_eval.astype("float32")
            xtr_cp_eval = xtr_cp_eval[sort_idx, :]
            x_cp_eval = xtr_cp_eval

    # Invokes super sampler and then sort its indexes
    class CustomBatchSampler(BatchSampler):
        def __iter__(self):
            for res in super().__iter__():
                yield sorted(res)
    batch_sampler = CustomBatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=True)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    model.to(device)
    # Model compilation
    optimizer = torch.optim.Adam(model.parameters(), lr=10**lr_exp, weight_decay=10**weight_decay_exp)
    loss_mse = torch.nn.MSELoss()

    for n in range(n_epochs):
        model.train()
        loss_ac = 0
        loss_mse_ac = 0
        loss_neg_log_ac = 0
        pbar = tqdm(enumerate(dataloader), disable=not verbose)
        #from ipdb import set_trace; set_trace()
        for idx, batch in pbar:
            if xtr_cp is None:
                x, e, t = batch
                decoded, risk_pred = model(x)
            else:
                x, e, t, x_cp = batch
                decoded, risk_pred = model(x, x_cp)
            mse_loss = loss_mse(x, decoded)
            neg_log = negative_log_likelihood(e, risk_pred)
            loss = mse_loss * alpha + (1 - alpha) * neg_log
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ac += loss.item()
            loss_mse_ac += mse_loss.item()
            loss_neg_log_ac += neg_log.item()
            pbar.set_description(
                'Epoch [{epoch}/{epochs}] :: Loss {loss:.4f}, MSE loss {mse_loss:.4f}, Neg Log {neg_log:.4f}'.format(
                    epoch=n + 1,
                    epochs=n_epochs,
                    loss=loss_ac / (idx + 1),
                    mse_loss=loss_mse_ac / (idx + 1),
                    neg_log=loss_neg_log_ac / (idx + 1),
                )
            )

        # run early stopping evaluation
        if xtr_eval is not None:
            model.eval()

            if xtr_cp_eval is None:
                _, risk_pred = model.predict(x_eval)
            else:
                _, risk_pred = model.predict(x_eval, x_cp_eval)
            eval_cindex = concordance_index(t_eval, -risk_pred, e_eval)

            model.loss_history_validation.append(eval_cindex)
            if eval_cindex >= model.best_cindex_val:
                model.best_cindex_val = eval_cindex
                best_state_dict = model.state_dict()
                best_state_dict = deepcopy(best_state_dict)
                es_tries = 0
                if verbose:
                    print("This is the lowest validation loss",
                          "so far.")
            else:
                es_tries += 1


            if es_tries == es_patience // 3 or es_tries == es_patience // 3 * 2:
                if verbose:
                    print("No improvement for", es_tries, "tries")
                    print("Decreasing learning rate by half")
                    print("Restarting from best route.")
                optimizer.param_groups[0]['lr'] *= 0.5
                model.load_state_dict(best_state_dict)
                model.loss_history_validation.append('Reduce lr')
            elif es_tries >= es_patience:
                model.load_state_dict(best_state_dict)
                if verbose:
                    print(
                        "Validation loss did not improve after",
                        es_patience,
                        "tries. Stopping"
                    )
                break


    return model
