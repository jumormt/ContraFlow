from torch import nn
from omegaconf import DictConfig
import torch
from src.datas.samples import MethodSampleBatch
from src.models.modules.attention import LocalAttention
from typing import Optional, Dict
from pytorch_lightning import LightningModule
from src.models.modules.flow_encoders import FlowHYBRIDEncoder, FlowLSTMEncoder, FlowBERTEncoder, FlowGNNEncoder
from torch.optim import Adam, SGD, Adamax, RMSprop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch.nn.functional as F
from src.metrics import Statistic
from torch.nn import TransformerEncoder, TransformerEncoderLayer, parameter
from torch_geometric.data import Batch
from src.utils import cut_lower_embeddings


class VulDetectModel(LightningModule):
    r"""vulnerability detection model to detect vulnerability at method-level and interpret at line-level
    This implementation is based on self-attention and attention mechanism

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary, e.g. tokenizer.get_vocab_size()
        pad_idx (int): the index of padding token, e.g., tokenizer.token_to_id(PAD)
        pretrain (str): path of .ckpt pre-trained model
    """

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
                 pretrain: Optional[str] = None,
                 pretrain_gnn: Optional[str] = None):
        super.__init__()
        self.__pretrain = pretrain
        self.__pretrain_gnn = pretrain_gnn
        hidden_size = config.encoder.flow_hidden_size
        if pretrain is not None:
            print("Use pretrained weights for vulnerability detection model")
            state_dict = torch.load(pretrain)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {
                k.removeprefix("_encoder."): v
                for k, v in state_dict.items() if k.startswith("_encoder.")
            }
            self._encoder.load_state_dict(state_dict)
        else:
            print("No pre-trained weights for vulnerability detection model")
            if config.encoder.name == "LSTM":
                self._encoder = FlowLSTMEncoder(config.encoder,
                                                vocabulary_size, pad_idx)
            elif config.encoder.name == "BERT":
                self._encoder = FlowBERTEncoder(config.encoder, pad_idx)
            elif config.encoder.name == "GNN":
                self._encoder = FlowGNNEncoder(config.encoder, vocabulary_size,
                                               pad_idx, pretrain_gnn)
            elif config.encoder.name == "HYBRID":
                self._encoder = FlowHYBRIDEncoder(config.encoder,
                                                  vocabulary_size, pad_idx,
                                                  pretrain_gnn)
            else:
                raise ValueError(
                    f"Cant find encoder model: {config.encoder.name}")

        # self-attention
        encoder_layers = TransformerEncoderLayer(hidden_size,
                                                 config.nhead,
                                                 hidden_size,
                                                 config.self_attn_dropout,
                                                 batch_first=True)
        self.__transformer_encoder = TransformerEncoder(
            encoder_layers, config.nlayers)
        self.__flow_attn = LocalAttention(hidden_size)
        # hidden layers
        layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        if config.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.n_hidden_layers})")
        for _ in range(config.n_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.__hidden_layers = nn.Sequential(*layers)

        self.__classifier = nn.Linear(hidden_size, config.n_classes)

    def forward(self, batch: Batch, statements: torch.Tensor,
                statements_per_value_flow: torch.Tensor,
                value_flow_per_label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            statements (Tensor): [seq len; total n_statements]
            statements_per_value_flow (Tensor): [total n_flow]
            value_flow_per_label (Tensor): [n_method]

        Returns: classifier results: [n_method; n_classes]
        """
        # [total n_flow; flow_hidden_size]
        if self.__config.encoder.name in ["LSTM", "BERT"]:
            value_flow_embeddings = self._encoder(statements,
                                                  statements_per_value_flow)
        elif self.__config.encoder.name == "GNN":
            value_flow_embeddings = self._encoder(batch,
                                                  statements_per_value_flow)
        elif self.__config.encoder.name == "HYBRID":
            value_flow_embeddings = self._encoder(batch, statements,
                                                  statements_per_value_flow)
        else:
            raise ValueError(
                f"Cant find encoder model: {self.__config.encoder.name}")
        # [total n_flow; max flow n_statements]
        flow_attn_weights, _ = self._encoder.get_flow_attention_weights()
        # [n_method; max method n_flow; max flow n_statements]
        self.__value_flow_attn_weights, _ = cut_lower_embeddings(
            flow_attn_weights, value_flow_per_label)

        # [n_method; max method n_flow; flow_hidden_size], [n_method; max method n_flow]
        method_flows_embeddings, method_flows_attn_mask = cut_lower_embeddings(
            value_flow_embeddings, value_flow_per_label)
        # [n_method; max method n_flow; flow_hidden_size]
        method_flows_embeddings = self.__transformer_encoder(
            method_flows_embeddings,
            mask=None,
            src_key_padding_mask=method_flows_attn_mask)
        # [n_method; max method n_flow; 1]
        self.__method_attn_weights = self.__flow_attn(method_flows_embeddings,
                                                      method_flows_attn_mask)
        # [n_method; flow_hidden_size]
        method_embeddings = torch.bmm(
            self.__method_attn_weights.transpose(1, 2),
            method_flows_embeddings).squeeze(1)
        hiddens = self.__hidden_layers(method_embeddings)
        # [n_method; n_classes]
        return self.__classifier(hiddens)

    def get_attention_weights(self):
        """get the attention scores of value flows and statements

        Returns:
            : [n_method, max n_flow] the importance of value flows
            : [n_method, max n_flow; max flow n_statements] the importance of statements on each value flow
        """
        return self.__method_attn_weights, self.__value_flow_attn_weights

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

    def training_step(self, batch: MethodSampleBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_method; n_classes]
        logits = self(batch.ast_graphs, batch.statements,
                      batch.statements_per_value_flow,
                      batch.value_flow_per_label)
        loss = F.cross_entropy(logits, batch.labels)
        # if pretrain is not None, we do not add loss
        if self.__pretrain is None and self.__config.encoder.name in [
                "GNN", "HYBRID"
        ] and self.__pretrain_gnn is None:
            if self.__config.encoder.name == "GNN":

                loss = loss + 4 * self._encoder._gnn_encoder.get_att_loss()
            else:
                loss = loss + 4 * self._encoder.__gnn_encoder._gnn_encoder.get_att_loss(
                )

        result: Dict = {"train/loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)

            method_attn_weights, value_flow_attn_weights = self.get_attention_weights(
            )
            true_positives_slice = (batch.labels == 1) & (preds == 1)
            btp_metric = statistic.calc_btp_metrics(
                true_positives_slice, batch.statements_idxes, batch.flaws,
                method_attn_weights, value_flow_attn_weights, "train")
            result.update(btp_metric)
            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train/f1"],
                     prog_bar=True,
                     logger=False)
        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: MethodSampleBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_method; n_classes]
        logits = self(batch.ast_graphs, batch.statements,
                      batch.statements_per_value_flow,
                      batch.value_flow_per_label)
        loss = F.cross_entropy(logits, batch.labels)
        # if pretrain is not None, we do not add loss
        if self.__pretrain is None and self.__config.encoder.name in [
                "GNN", "HYBRID"
        ] and self.__pretrain_gnn is None:
            if self.__config.encoder.name == "GNN":

                loss = loss + 4 * self._encoder._gnn_encoder.get_att_loss()
            else:
                loss = loss + 4 * self._encoder.__gnn_encoder._gnn_encoder.get_att_loss(
                )

        result: Dict = {"val/loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="val")
            result.update(batch_metric)
            method_attn_weights, value_flow_attn_weights = self.get_attention_weights(
            )
            true_positives_slice = (batch.labels == 1) & (preds == 1)
            btp_metric = statistic.calc_btp_metrics(
                true_positives_slice, batch.statements_idxes, batch.flaws,
                method_attn_weights, value_flow_attn_weights, "val")
            result.update(btp_metric)
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: MethodSampleBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_method; n_classes]
        logits = self(batch.ast_graphs, batch.statements,
                      batch.statements_per_value_flow,
                      batch.value_flow_per_label)
        loss = F.cross_entropy(logits, batch.labels)
        # if pretrain is not None, we do not add loss
        if self.__pretrain is None and self.__config.encoder.name in [
                "GNN", "HYBRID"
        ] and self.__pretrain_gnn is None:
            if self.__config.encoder.name == "GNN":

                loss = loss + 4 * self._encoder._gnn_encoder.get_att_loss()
            else:
                loss = loss + 4 * self._encoder.__gnn_encoder._gnn_encoder.get_att_loss(
                )


        result: Dict = {"test/loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)
            method_attn_weights, value_flow_attn_weights = self.get_attention_weights(
            )
            true_positives_slice = (batch.labels == 1) & (preds == 1)
            btp_metric = statistic.calc_btp_metrics(
                true_positives_slice, batch.statements_idxes, batch.flaws,
                method_attn_weights, value_flow_attn_weights, "test")
            result.update(btp_metric)

        return {"loss": loss, "statistic": statistic}

    # ========== EPOCH END ==========
    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, group: str):
        log = self._prepare_epoch_end_log(step_outputs, group)
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in step_outputs])
        log.update(statistic.calculate_metrics(group))
        log.update(statistic.mean_BTP(group))
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
