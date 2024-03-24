import torch
from torch import nn
import pytorch_lightning as pl

EPSILON = 1e-8


class WeibullLoss(pl.LightningModule):
    def __init__(self, clip_prob: float = 1e-6):
        super().__init__()
        self.clip_prob = clip_prob

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y, u = torch.split(y_true, 1, dim=1)  # time to event (discrete) / censoring

        loglikelihoods = self.loglik_discrete(y, u, y_pred[:, 0], y_pred[:, 1])

        loglikelihoods = torch.clamp(
            loglikelihoods,
            torch.log(torch.tensor(self.clip_prob)),
            torch.log(torch.tensor(1 - self.clip_prob)),
        )
        loss = -torch.mean(loglikelihoods)
        return loss

    def loglik_discrete(
        self, y: torch.Tensor, u: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        hazard0 = torch.pow((y + EPSILON) / a, b)
        hazard1 = torch.pow((y + 1.0) / a, b)

        loglikelihoods = (
            u * torch.log(torch.exp(hazard1 - hazard0) - (1.0 - EPSILON)) - hazard1
        )
        return loglikelihoods


class MLP(pl.LightningModule):
    def __init__(self, lr: float, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.lr = lr
        self.loss = WeibullLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.loss(y, y_hat)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        self.common_step(batch=batch, batch_idx=batch_idx)

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        self.common_step(batch=batch, batch_idx=batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    pass
