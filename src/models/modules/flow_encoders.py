from torch import nn
from omegaconf import DictConfig
import torch
from src.models.modules.attention import LocalAttention
import numpy
from src.utils import cut_lower_embeddings
from transformers import RobertaModel, RobertaConfig
from src.models.modules.common_layers import FlowGRULayer, SuperGATConvEncoder, linear_after_attn
from torch_geometric.data import Batch
from typing import Optional


class FlowLSTMEncoder(nn.Module):
    r"""The value flow encoder to transform a value flow into a compact vector.
    This implementation is based on attention-based RNNs

    Args:
        config (DictConfig): configuration for the encoder
        vocabulary_size (int): the size of vacabulary, e.g. tokenizer.get_vocab_size()
        pad_idx (int): the index of padding token, e.g., tokenizer.token_to_id(PAD)
    """
    __negative_value = -numpy.inf

    def __init__(self, config: DictConfig, vocabulary_size: int, pad_idx: int):
        super().__init__()
        self.__pad_idx = pad_idx
        self.__st_embedding = nn.Embedding(vocabulary_size,
                                           config.embed_dim,
                                           padding_idx=pad_idx)
        torch.nn.init.xavier_normal_(self.__st_embedding.weight.data)

        self.__st_blstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.st_hidden_size,
            num_layers=config.st_num_layers,
            bidirectional=config.st_use_bi_rnn,
            dropout=config.st_dropout if config.st_num_layers > 1 else 0,
        )
        self.__dropout_st_blstm = nn.Dropout(config.st_dropout)
        self.__st_att = LocalAttention(2 * config.st_hidden_size)
        self.__st_hidden = linear_after_attn(config.st_hidden_size,
                                             config.st_hidden_size,
                                             config.activation)
        self.__flow_gru = FlowGRULayer(input_dim=config.st_hidden_size,
                                       out_dim=config.flow_hidden_size,
                                       num_layers=config.flow_num_layers,
                                       use_bi=config.flow_use_bi_rnn,
                                       dropout=config.flow_dropout,
                                       activation=config.activation)

    def forward(self, statements: torch.Tensor,
                statements_per_label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            statements (Tensor): [seq len; total n_statements]
            statements_per_label (Tensor): [n_flow]

        Returns: flow_embedding: [n_flow; flow_hidden_size]
        """
        # [seq len; total n_statements, embed dim]
        embeds = self.__st_embedding(statements)
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(
                statements == self.__pad_idx, dim=0)
            first_pad_pos[~is_contain_pad_id] = statements.shape[
                0]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,
                                                           descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        embeds = embeds[:, sort_indices]
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeds, sorted_path_lengths)
        st_hiddens, (_, _) = self.__st_blstm(packed_embeddings)
        # [seq len; total n_statements, 2 * st hidden size]
        st_hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            st_hiddens, total_length=statements.size(0))
        # [total n_statements, seq len; 2 * st hidden size]
        st_hiddens_bf = self.__dropout_st_blstm(st_hiddens.permute(
            1, 0, 2))[reverse_sort_indices]

        # [total n_statements, seq len]
        st_hidden_att_mask = st_hiddens.new_zeros(st_hiddens.size(1),
                                                  st_hiddens.size(0))
        st_hidden_att_mask[statements.transpose(0, 1) ==
                           self.__pad_idx] = self.__negative_value
        # [total n_statements; seq len; 1]
        st_attn_weights = self.__st_att(st_hiddens_bf, st_hidden_att_mask)
        # [total n_statements, st hidden size]
        statements_embeddings = self.__st_hidden(
            torch.bmm(st_attn_weights.transpose(1, 2),
                      st_hiddens_bf).squeeze(1))

        # [n_flow; max flow n_statements; seq len]
        self.__flow_statments_weights, _ = cut_lower_embeddings(
            st_attn_weights.squeeze(2), statements_per_label,
            self.__negative_value)
        # [n_flow, flow hidden size]
        flow_embeddings, self.__flow_attn_weights = self.__flow_gru(
            statements_embeddings, statements_per_label)
        return flow_embeddings

    def get_flow_attention_weights(self):
        """get the attention scores of statements and tokens

        Returns:
            : [n_flow; max flow n_statements] the importance of statements on each value flow
            : [n_flow; max flow n_statements; seq len] the importance of tokens on each statement on each value flow
        """
        return self.__flow_attn_weights.squeeze(
            2), self.__flow_statments_weights


class FlowBERTEncoder(nn.Module):
    r"""The value flow encoder to transform a value flow into a compact vector.
    This implementation is based on CodeBert (RobertaModel)

    Args:
        config (DictConfig): configuration for the encoder
        pad_idx (int): the index of padding token, e.g., tokenizer.pad_token_id
    """
    def __init__(self, config: DictConfig, pad_idx: int):
        super().__init__()
        self.__pad_idx = pad_idx
        model_name = "microsoft/codebert-base"
        bert_config = RobertaConfig.from_pretrained(model_name)
        self.__st_encoder = RobertaModel.from_pretrained(model_name,
                                                         config=bert_config)
        self.__flow_gru = FlowGRULayer(input_dim=bert_config.hidden_size,
                                       out_dim=config.flow_hidden_size,
                                       num_layers=config.flow_num_layers,
                                       use_bi=config.flow_use_bi_rnn,
                                       dropout=config.flow_dropout,
                                       activation=config.activation)

    def forward(self, statements: torch.Tensor,
                statements_per_label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            statements (Tensor): [seq len; total n_statements]
            statements_per_label (Tensor): [n_flow]

        Returns: flow_embedding: [n_flow; flow_hidden_size]
        """

        statements = statements.transpose(0, 1)
        # [total n_statements, seq len] "0" - masked, "1" - not masked
        st_mask = statements.new_ones(statements.size(0), statements.size(1))
        st_mask[statements == self.__pad_idx] = 0
        # # [total n_statements; seq len; 1]
        # st_attn_weights = self.__st_att(st_hidden, st_mask)
        # # [total n_statements, bert hidden size]
        # statements_embeddings = self.__st_hidden(
        #     torch.bmm(st_attn_weights.transpose(1, 2), st_hidden).squeeze(1))

        # # [n_flow; max flow n_statements; seq len]
        # self.__flow_statments_weights, _ = cut_lower_embeddings(
        #     st_attn_weights.squeeze(2), statements_per_label,
        #     self.__negative_value)

        # [total n_statements; bert hidden size]
        statements_embeddings = self.__st_encoder(statements, st_mask)[1]
        # [n_flow, flow hidden size]
        flow_embeddings, self.__flow_attn_weights = self.__flow_gru(
            statements_embeddings, statements_per_label)
        return flow_embeddings

    def get_flow_attention_weights(self):
        """get the attention scores of statements and tokens

        Returns:
            : [n_flow; max flow n_statements] the importance of statements on each value flow
        """
        return self.__flow_attn_weights.squeeze(2), None


class FlowGNNEncoder(nn.Module):
    r"""The value flow encoder to transform a value flow into a compact vector.
    This implementation is based on GNN (GINEConv)

    Args:
        config (DictConfig): configuration for the encoder
        pad_idx (int): the index of padding token, e.g., tokenizer.pad_token_id
    """
    def __init__(self,
                 config: DictConfig,
                 vocabulary_size: int,
                 pad_idx: int,
                 pretrain: Optional[str] = None):
        super().__init__()
        # self.__gnn_encoder = GINEConvEncoder(config, vocabulary_size, pad_idx)
        if pretrain is not None:
            print("Use pretrained weights for flow gnn encoder")
            state_dict = torch.load(pretrain)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {
                k.removeprefix("_gnn_encoder."): v
                for k, v in state_dict.items() if k.startswith("_gnn_encoder.")
            }
            self._encoder.load_state_dict(state_dict)
        else:
            print("No pretrained weights for flow gnn encoder")
            self._gnn_encoder = SuperGATConvEncoder(config, vocabulary_size,
                                                    pad_idx)
        self.__flow_gru = FlowGRULayer(input_dim=config.ast.hidden_dim,
                                       out_dim=config.flow_hidden_size,
                                       num_layers=config.flow_num_layers,
                                       use_bi=config.flow_use_bi_rnn,
                                       dropout=config.flow_dropout,
                                       activation=config.activation)

    def forward(self, batch: Batch,
                statements_per_label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            batch (Tensor): [total n_statements (Data)]
            statements_per_label (Tensor): [n_flow]

        Returns: flow_embedding: [n_flow; flow_hidden_size]
        """

        # [total n_statements; ast hidden dim]
        statements_embeddings = self._gnn_encoder(batch)
        # [n_flow, flow hidden size]
        flow_embeddings, self.__flow_attn_weights = self.__flow_gru(
            statements_embeddings, statements_per_label)
        return flow_embeddings

    def get_flow_attention_weights(self):
        """get the attention scores of statements and tokens

        Returns:
            : [n_flow; max flow n_statements] the importance of statements on each value flow
        """
        return self.__flow_attn_weights.squeeze(2), None


class FlowHYBRIDEncoder(nn.Module):
    r"""The value flow encoder to transform a value flow into a compact vector.
    This implementation is based on LSTM and CodeBert (RobertaModel)

    Args:
        config (DictConfig): configuration for the encoder
        vocabulary_size (int): the size of vacabulary, e.g. tokenizer.get_vocab_size()
        pad_idx (int): the index of padding token, e.g., tokenizer.token_to_id(PAD)
    """
    def __init__(self,
                 config: DictConfig,
                 vocabulary_size: int,
                 pad_idx: int,
                 pretrain: Optional[str] = None):
        super().__init__()
        # we can use the tokenizer of bert
        self.__lstm_encoder = FlowLSTMEncoder(config, vocabulary_size, pad_idx)
        self.__bert_encoder = FlowBERTEncoder(config, vocabulary_size, pad_idx)
        self.__gnn_encoder = FlowGNNEncoder(config, vocabulary_size, pad_idx,
                                            pretrain)
        self.__fuse_layer = nn.Linear(3 * config.flow_hidden_size,
                                      config.flow_hidden_size)

    def forward(self, batch: Batch, statements: torch.Tensor,
                statements_per_label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            statements (Tensor): [seq len; total n_statements]
            batch (Batch): [total n_statements (Data)]
            statements_per_label (Tensor): [n_flow]

        Returns: flow_embedding: [n_flow; flow_hidden_size]
        """
        # [n_flows; flow_hidden_size]
        lstm_flow_embeddings = self.__lstm_encoder(statements,
                                                   statements_per_label)
        bert_flow_embeddings = self.__bert_encoder(statements,
                                                   statements_per_label)
        gnn_flow_embeddings = self.__gnn_encoder(batch, statements_per_label)
        flow_embeddings = self.__fuse_layer(
            torch.cat([
                lstm_flow_embeddings, bert_flow_embeddings, gnn_flow_embeddings
            ],
                      dim=1))

        return flow_embeddings

    def get_flow_attention_weights(self):
        """get the attention scores of statements and tokens

        Returns:
            : [n_flow; max flow n_statements] the importance of statements on each value flow
            : [n_flow; max flow n_statements; seq len] the importance of tokens on each statement on each value flow
        """
        return self.__lstm_encoder.__flow_attn_weights.squeeze(
            2), self.__lstm_encoder.__flow_statments_weights
