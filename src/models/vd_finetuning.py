from torch import nn
from omegaconf import DictConfig
import torch
from datas.datastructures import MethodSampleBatch
from src.models.modules.attention import LocalAttention
import numpy
from typing import List, Tuple, Optional, Dict
from pytorch_lightning import LightningModule
from src.models.modules.flow_encoder import FlowEncoder
from torch.optim import Adam, SGD, Adamax, RMSprop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch.nn.functional as F
from src.metrics import Statistic
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
                 pretrain: Optional[str] = None):
        super.__init__()
        hidden_size = config.encoder.flow_hidden_size
        if pretrain is not None:
            print("Use pretrained weights for sequence generating model")
            state_dict = torch.load(pretrain)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {
                k.removeprefix("_encoder."): v
                for k, v in state_dict.items() if k.startswith("_encoder.")
            }
            self._encoder.load_state_dict(state_dict)
        else:
            print("No pre-trained weights for sequence generating model")
            self._encoder = FlowEncoder(config.encoder, vocabulary_size,
                                        pad_idx)
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

    def forward(self, statements: torch.Tensor,
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
        value_flow_embeddings = self._encoder(statements,
                                              statements_per_value_flow)
        # [total n_flow; max flow n_statements]
        flow_attn_weights, _ = self._encoder.get_flow_attention_weights()
        # [n_method; max method n_flow; max flow n_statements]
        self.__value_flow_attn_weights, _ = self._cut_value_flow_embeddings(
            flow_attn_weights, value_flow_per_label)

        # [n_method; max method n_flow; flow_hidden_size], [n_method; max method n_flow]
        method_flows_embeddings, method_flows_attn_mask = self._cut_value_flow_embeddings(
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

    def _segment_sizes_to_slices(self, sizes: torch.Tensor) -> List:
        cum_sums = numpy.cumsum(sizes.cpu())
        start_of_segments = numpy.append([0], cum_sums[:-1])
        return [
            slice(start, end)
            for start, end in zip(start_of_segments, cum_sums)
        ]

    def _cut_value_flow_embeddings(
            self,
            value_flow_embeddings: torch.Tensor,
            value_flow_per_label: torch.Tensor,
            mask_value: float = -1e9) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cut value flow embeddings into method embeddings

        Args:
            value_flow_embeddings (Tensor): [total n_flow; units]
            value_flow_per_label (Tensor): [n_method]
            mask_value (float): -inf

        Returns: [n_method; max method n_flow; units], [n_method; max method n_flow]
        """
        batch_size = len(value_flow_per_label)
        max_context_len = max(value_flow_per_label)

        method_flows_embeddings = value_flow_embeddings.new_zeros(
            (batch_size, max_context_len, value_flow_embeddings.shape[-1]))
        method_flows_attn_mask = value_flow_embeddings.new_zeros(
            (batch_size, max_context_len))

        statments_slices = self._segment_sizes_to_slices(value_flow_per_label)
        for i, (cur_slice, cur_size) in enumerate(
                zip(statments_slices, value_flow_per_label)):
            method_flows_embeddings[
                i, :cur_size] = value_flow_embeddings[cur_slice]
            method_flows_attn_mask[i, cur_size:] = mask_value

        return method_flows_embeddings, method_flows_attn_mask

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict:

        optimizer = self._get_optimizer(
            self.__config.hyper_parameters.optimizer)(
                self._get_parameters(),
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
        logits = self(batch.statements, batch.statements_per_value_flow,
                      batch.value_flow_per_label)
        loss = F.cross_entropy(logits, batch.labels)
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
            # TODO: interpretation evaluation

            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train/f1"],
                     prog_bar=True,
                     logger=False)
        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: MethodSampleBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_method; n_classes]
        logits = self(batch.statements, batch.statements_per_value_flow,
                      batch.value_flow_per_label)
        loss = F.cross_entropy(logits, batch.labels)
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

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: MethodSampleBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_method; n_classes]
        logits = self(batch.statements, batch.statements_per_value_flow,
                      batch.value_flow_per_label)
        loss = F.cross_entropy(logits, batch.labels)
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
        log.update(
            Statistic.union_statistics([
                out["statistic"] for out in step_outputs
            ]).calculate_metrics(group))
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
