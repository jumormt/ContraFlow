from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from src.models.modules.flow_encoders import FlowGNNEncoder, FlowLSTMEncoder, FlowBERTEncoder, FlowHYBRIDEncoder
from src.datas.samples import ValueFlowBatch
import torch
from typing import List, Iterator, Dict, Optional
from torch.nn import Parameter
from torch.optim import Adam, SGD, Adamax, RMSprop
from src.loss import NCE_loss
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch_geometric.data import Batch


class FlowCLPretraining(LightningModule):

    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    def __init__(self,
                 config: DictConfig,
                 vocabulary_size: int,
                 pad_idx: int,
                 pretrain_gnn: Optional[str] = None):
        super().__init__()
        self.__config = config
        self.__pretrain_gnn = pretrain_gnn
        if config.encoder.name == "LSTM":
            self._encoder = FlowLSTMEncoder(config.encoder, vocabulary_size,
                                            pad_idx)
        elif config.encoder.name == "BERT":
            self._encoder = FlowBERTEncoder(config.encoder, pad_idx)
        elif config.encoder.name == "GNN":
            self._encoder = FlowGNNEncoder(config.encoder, vocabulary_size,
                                           pad_idx, pretrain_gnn)
        elif config.encoder.name == "HYBRID":
            self._encoder = FlowHYBRIDEncoder(config.encoder, vocabulary_size,
                                              pad_idx, pretrain_gnn)
        else:
            raise ValueError(f"Cant find encoder model: {config.encoder.name}")

    def forward(self, batch: Batch, statements: torch.Tensor,
                statements_per_label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            statements (Tensor): [seq len; total n_statements]
            statements_per_label (Tensor): [n_flow]

        Returns: flow_embedding: [n_flow; flow_hidden_size]
        """
        if self.__config.encoder.name in ["LSTM", "BERT"]:
            return self._encoder(statements, statements_per_label)
        elif self.__config.encoder.name == "GNN":
            return self._encoder(batch, statements_per_label)
        elif self.__config.encoder.name == "HYBRID":
            return self._encoder(batch, statements, statements_per_label)
        else:
            raise ValueError(
                f"Cant find encoder model: {self.__config.encoder.name}")

    def _get_parameters(self) -> List[Iterator[Parameter]]:
        return [self._encoder.parameters()]

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

    def training_step(self, batch: ValueFlowBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_flow; flow_hidden_size]
        embeddings = self(batch.ast_graphs, batch.statements,
                          batch.statements_per_label)
        loss = NCE_loss(embeddings, batch.sequences)
        if self.__config.encoder.name in ["GNN", "HYBRID"
                                          ] and self.__pretrain_gnn is None:
            if self.__config.encoder.name == "GNN":

                loss = loss + 4 * self._encoder._gnn_encoder.get_att_loss()
            else:
                loss = loss + 4 * self._encoder.__gnn_encoder._gnn_encoder.get_att_loss(
                )
        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch: ValueFlowBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_flow; flow_hidden_size]
        embeddings = self(batch.ast_graphs, batch.statements,
                          batch.statements_per_label)
        loss = NCE_loss(embeddings, batch.sequences)
        if self.__config.encoder.name in ["GNN", "HYBRID"
                                          ] and self.__pretrain_gnn is None:
            if self.__config.encoder.name == "GNN":

                loss = loss + 4 * self._encoder._gnn_encoder.get_att_loss()
            else:
                loss = loss + 4 * self._encoder.__gnn_encoder._gnn_encoder.get_att_loss(
                )
        return loss

    def test_step(self, batch: ValueFlowBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_flow; flow_hidden_size]
        embeddings = self(batch.ast_graphs, batch.statements,
                          batch.statements_per_label)
        loss = NCE_loss(embeddings, batch.sequences)
        if self.__config.encoder.name in ["GNN", "HYBRID"
                                          ] and self.__pretrain_gnn is None:
            if self.__config.encoder.name == "GNN":

                loss = loss + 4 * self._encoder._gnn_encoder.get_att_loss()
            else:
                loss = loss + 4 * self._encoder.__gnn_encoder._gnn_encoder.get_att_loss(
                )
        return loss

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
