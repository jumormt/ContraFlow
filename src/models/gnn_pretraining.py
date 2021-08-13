from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from src.models.modules.common_layers import SuperGATConvEncoder
import torch
from typing import List, Iterator, Dict
from torch.nn import Parameter
from torch.optim import Adam, SGD, Adamax, RMSprop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch_geometric.data import Batch


class GNNPretraining(LightningModule):

    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    def __init__(self, config: DictConfig, vocabulary_size: int, pad_idx: int):
        super().__init__()
        self.__config = config

        self._gnn_encoder = SuperGATConvEncoder(config, vocabulary_size,
                                                pad_idx)

    def forward(self, batch: Batch) -> torch.Tensor:
        """

        Args:
            batch [Batch]: AST Datas

        Returns: loss
        """
        self._gnn_encoder(batch)
        return self._gnn_encoder.get_att_loss()

    def _get_parameters(self) -> List[Iterator[Parameter]]:
        return [self._gnn_encoder.parameters()]

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict:
        parameters = self._get_parameters()
        optimizer = self._get_optimizer(
            self.__config.hyper_parameters.optimizer)(
                [{"params": p} for p in parameters],
                self.__config.hyper_parameters.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
            **epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)

    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def training_step(self, batch: Batch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # []
        loss = self(batch)
        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch: Batch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        return self(batch)

    def test_step(self, batch: Batch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        return self(batch)

    # ========== EPOCH END ==========

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, step: str):
        log = self._prepare_epoch_end_log(step_outputs, step)
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
