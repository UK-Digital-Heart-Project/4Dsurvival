from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class TorchModel(torch.nn.Module):
    def predict(self, x: np.ndarray, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(x).cuda().float()
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.eval()
        decodeds = []
        risks = []
        for x in dataloader:
            decoded, risk_pred = self(x[0])
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

class BaselineBNAutoencoder2(TorchModel):
    def __init__(self, input_shape: int, dropout: float, num_ae_units1: int, num_ae_units2: int):
        super().__init__()
        num_ae_units1 = round(num_ae_units1)
        num_ae_units2 = round(num_ae_units2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape, num_ae_units1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(num_ae_units1, num_ae_units2),
            torch.nn.BatchNorm1d(num_ae_units2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units2, num_ae_units2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_ae_units1),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units1, input_shape)
        )
        self.risk_regressor = torch.nn.Sequential(
            torch.nn.Linear(num_ae_units2, num_ae_units2//2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_ae_units2//2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_ae_units2//2, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        risk_pred = self.risk_regressor(encoded)
        decoded = self.decoder(encoded)
        return decoded, risk_pred


def model_factory(model_name: str, **kwargs) -> TorchModel:
    # Before defining network architecture, clear current computation graph (if one exists)
    torch.cuda.empty_cache()
    if model_name == "baseline_autoencoder":
        model = BaselineAutoencoder(**kwargs)
    elif model_name == "baseline_bn_autoencoder":
        model = BaselineBNAutoencoder(**kwargs)
    elif model_name == "baseline_bn_autoencoder_2":
        model = BaselineBNAutoencoder2(**kwargs)
    else:
        raise ValueError("Model name {} has not been implemented.".format(model_name))
    return model
