import torch
from torch import nn
import pytorch_lightning as pl

EPSILON = 1e-7


class WeibullLoss(pl.LightningModule):
    def __init__(self, clip_prob: float = 1e-6):
        super().__init__()
        self.clip_prob = clip_prob

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y, u = torch.split(y_true, 1, dim=1)  # time to event (discrete) / censoring

        hazard0 = torch.pow(
            (y.squeeze(dim=-1) + EPSILON) / (y_pred[:, 0]), y_pred[:, 1]
        )
        hazard1 = torch.pow((y.squeeze(dim=-1) + 1.0) / (y_pred[:, 0]), y_pred[:, 1])

        loglikelihoods = (
            u.squeeze(dim=-1)
            * torch.log(torch.exp(hazard1 - hazard0) - (1.0 - EPSILON))
            - hazard1
        )

        loglikelihoods = torch.clamp(
            loglikelihoods,
            torch.log(torch.tensor(self.clip_prob)),
            torch.log(torch.tensor(1 - self.clip_prob)),
        )
        loss = -torch.mean(loglikelihoods)

        return loss


class MLP(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        input_size: int = 512,
        hidden_size: int | tuple = 256,
        hparams: dict | None = None,
    ):
        super().__init__()
        if hparams:
            self.save_hyperparameters(hparams)
        if isinstance(hidden_size, int):
            self.hidden_sizes = [hidden_size]
        elif isinstance(hidden_size, tuple) or isinstance(hidden_size, list):
            self.hidden_sizes = list(hidden_size)

        layers = []
        prev_size = input_size
        for size in self.hidden_sizes:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            layers.append(nn.Linear(prev_size, size))

            prev_size = size

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(prev_size, 2))
        layers.append(nn.Softplus())

        self.layers = nn.Sequential(*layers)
        self.lr = lr
        self.loss = WeibullLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        # x = x.view(x.size(0), -1)

        y_hat = self.layers(x)
        loss = self.loss(y, y_hat)

        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._common_step(batch=batch, batch_idx=batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._common_step(batch=batch, batch_idx=batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
