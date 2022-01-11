from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class TorchModel(torch.nn.Module):
    def predict(self, x: np.ndarray, x_cp: np.ndarray = None,
        batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x).to(device).float()
        dataset = [x]
        if x_cp is not None:
            x_cp = torch.from_numpy(x_cp).to(device).float()
            dataset.append(x_cp)
        dataset = TensorDataset(*dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.eval()
        decodeds = []
        risks = []
        for batch in dataloader:
            if x_cp is None:
                xb, = batch
                decoded, risk_pred = self(xb)
            else:
                xb, xb_cp = batch
                decoded, risk_pred = self(xb, xb_cp)
            decodeds.append(decoded)
            risks.append(risk_pred)
        decodeds = torch.cat(decodeds, dim=0)
        decodeds = decodeds.cpu().detach().numpy()
        risks = torch.cat(risks, dim=0)
        risks = risks.cpu().detach().numpy()
        return decodeds, risks


class BaselineAutoencoder(TorchModel):
    def __init__(self, input_shape: int, dropout: float, num_ae_units1: int, num_ae_units2: int):
        super().__init__()
        num_ae_units1 = round(num_ae_units1)
        num_ae_units2 = round(num_ae_units2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_shape, num_ae_units1),
            torch.nn.ReLU(),
            torch.nn.Linear(num_ae_units1, num_ae_units2),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units1),
            torch.nn.ReLU(),
            torch.nn.Linear(num_ae_units1, input_shape)
        )
        self.risk_regressor = torch.nn.Linear(num_ae_units2, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        risk_pred = self.risk_regressor(encoded)
        decoded = self.decoder(encoded)
        return decoded, risk_pred


class BaselineBNAutoencoder(TorchModel):
    def __init__(self, input_shape: int, dropout: float, num_ae_units1: int, num_ae_units2: int):
        super().__init__()
        num_ae_units1 = round(num_ae_units1)
        num_ae_units2 = round(num_ae_units2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_shape, num_ae_units1),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.ReLU(),
            torch.nn.Linear(num_ae_units1, num_ae_units2),
            torch.nn.BatchNorm1d(num_ae_units2),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units1),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.ReLU(),
            torch.nn.Linear(num_ae_units1, input_shape)
        )
        self.risk_regressor = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units2//2),
            torch.nn.BatchNorm1d(num_ae_units2//2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_ae_units2//2, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        risk_pred = self.risk_regressor(encoded)
        decoded = self.decoder(encoded)
        return decoded, risk_pred

class BaselineBNAutoencoderMC(TorchModel):
    def __init__(self, input_shape: int, dropout: float, num_ae_units1: int, num_ae_units2: int):
        super().__init__()
        num_ae_units1 = round(num_ae_units1)
        num_ae_units2 = round(num_ae_units2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape, num_ae_units1),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(num_ae_units1, num_ae_units2),
            torch.nn.BatchNorm1d(num_ae_units2),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units2, num_ae_units2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units1),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units1, input_shape)
        )
        self.risk_regressor = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units2//2),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_ae_units2//2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units2//2, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        risk_pred = self.risk_regressor(encoded)
        decoded = self.decoder(encoded)
        return decoded, risk_pred

class BaselineBNAutoencoderWithCP(TorchModel):
    def __init__(self, input_shape: int,
        dropout: float, num_ae_units1: int, num_ae_units2: int):
        super().__init__()
        num_ae_units1 = round(num_ae_units1)
        num_ae_units2 = round(num_ae_units2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape, num_ae_units1),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(num_ae_units1, num_ae_units2),
            torch.nn.BatchNorm1d(num_ae_units2),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units2, num_ae_units2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units1),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units1, input_shape)
        )
        self.risk_regressor = torch.nn.Sequential(
            torch.nn.LazyLinear(num_ae_units2//2),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_ae_units2//2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units2//2, 1)
        )

    def forward(self, x_mesh, x_cp):
        encoded_mesh = self.encoder(x_mesh)
        encoded_mesh_cp = torch.cat([encoded_mesh, x_cp], 1)
        risk_pred = self.risk_regressor(encoded_mesh_cp)
        decoded = self.decoder(encoded_mesh)
        return decoded, risk_pred


def model_factory(model_name: str, **kwargs) -> TorchModel:
    # Before defining network architecture, clear current computation graph (if one exists)
    torch.cuda.empty_cache()
    if model_name == "baseline_autoencoder":
        model = BaselineAutoencoder(**kwargs)
    elif model_name == "baseline_bn_autoencoder":
        model = BaselineBNAutoencoder(**kwargs)
    elif model_name == "baseline_bn_autoencoder_mc":
        model = BaselineBNAutoencoderMC(**kwargs)
    elif model_name == "baseline_bn_autoencoder_with_cp":
        model = BaselineBNAutoencoderWithCP(**kwargs)
    else:
        raise ValueError("Model name {} has not been implemented.".format(model_name))
    return model
