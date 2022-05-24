from lightningmodels.vision.segmentation import Segmenter
import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

@MODEL_REGISTRY
class SelfEnsemblingSegmenter(Segmenter):

    def __init__(self, reducer:str = "mean", *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert reducer in ["mean", "noisyor"]

        self.reducer = reducer

    def forward(self, x):
        x_1 = x
        x_2 = x.fliplr()
        x_3 = x.flipud()
        x_4 = x.fliplr().flipud()

        y_1 = self.model(x_1)
        y_2 = self.model(x_2).fliplr()
        y_3 = self.model(x_3).flipud()
        y_4 = self.model(x_4).fliplr().flipud()

        return torch.stack([y_1, y_2, y_3, y_4])

    def reduce(self, y_hats):
        if self.reducer == "noisyor":
            y_hat = 1 - torch.prod(1 - y_hats, dim=0)
        elif self.reducer == "mean":
            y_hat = torch.mean(y_hats, dim=0)
        else:
            raise ValueError(f"Expecting reducer to be one of mean, noisyor, found {self.reducer}")
        return y_hat

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hats = self(x)
        imbalance = y_hats.mean(dim=(1,2,3,4)).std()
        self.log("imbalance", imbalance, on_step=False, on_epoch=True, prog_bar=True)

        diversity = y_hats.std(axis=0)
        agree_false = y_hats.sum(axis=0) < 0.05
        agree_true = y_hats.sum(axis=0) > 3.95
        diversity_mask = torch.logical_not(torch.logical_or(agree_false, agree_true))
        aggregated_diversity = diversity[diversity_mask].mean()
        self.log("agree_false", agree_false.sum().float() / y.numel(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("agree_true", agree_true.sum().float() / y.numel(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("diversity", aggregated_diversity, on_step=True, on_epoch=True, prog_bar=True)

        y_hat = self.reduce(y_hats)

        background = torch.ones(
            [y_hat.size(dim=0), 1,
             y_hat.size(dim=2),
             y_hat.size(dim=3)],
            dtype=y_hat.dtype,
            device=y_hat.device) * 0.5

        y_hat_label = torch.concat([background, y_hat], dim=1).argmax(dim=1)
        y_label = torch.concat([background, y], dim=1).argmax(dim=1)
        self.train_f1(y_hat_label, y_label)
        self.log('train_f1', self.train_f1, prog_bar=True)

        return self.loss(y_hat, y)

    def validation_step(self, batch, _batch_idx):
        x, y = batch

        y_hats = self(x)

        imbalance = y_hats.mean(dim=(1,2,3,4)).std()
        self.log("imbalance", imbalance, on_step=False, on_epoch=True, prog_bar=True)

        diversity = y_hats.std(axis=0)
        agree_false = y_hats.sum(axis=0) < 0.05
        agree_true = y_hats.sum(axis=0) > 3.95
        diversity_mask = torch.logical_not(torch.logical_or(agree_false, agree_true))
        aggregated_diversity = diversity[diversity_mask].mean()
        self.log("agree_false", agree_false.sum().float() / y.numel(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("agree_true", agree_true.sum().float() / y.numel(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("diversity", aggregated_diversity, on_step=True, on_epoch=True, prog_bar=True)

        y_hat = self.reduce(y_hats)

        background = torch.ones(
            [y_hat.size(dim=0), 1,
             y_hat.size(dim=2),
             y_hat.size(dim=3)],
            dtype=y_hat.dtype,
            device=y_hat.device) * 0.5

        y_hat_label = torch.concat([background, y_hat], dim=1).argmax(dim=1)
        y_label = torch.concat([background, y], dim=1).argmax(dim=1)
        self.valid_f1(y_hat_label, y_label)

        self.log('valid_f1',
                 self.valid_f1,
                 prog_bar=True)


        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
