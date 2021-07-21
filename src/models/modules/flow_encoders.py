from numpy.lib.arraypad import pad
from torch import nn
from omegaconf import DictConfig
import torch
from src.models.modules.attention import LocalAttention
import numpy
from typing import Tuple
from transformers import RobertaModel, RobertaConfig
from src.utils import segment_sizes_to_slices
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, TopKPooling
from src.datas.datastructures import NodeType
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def linear_after_attn(in_dim: int, out_dim: int, activation: str) -> nn.Module:
    """Linear layers after attention

        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            activation (str): the name of activation function
        """
    # add drop out?
    return torch.nn.Sequential(
        torch.nn.Linear(2 * in_dim, 2 * in_dim),
        torch.nn.BatchNorm1d(2 * in_dim),
        get_activation(activation),
        torch.nn.Linear(2 * in_dim, out_dim),
    )


activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "lkrelu": nn.LeakyReLU(0.3)
}


def get_activation(activation_name: str) -> torch.nn.Module:
    if activation_name in activations:
        return activations[activation_name]
    raise KeyError(f"Activation {activation_name} is not supported")


def cut_statements_embeddings(
        statements_embeddings: torch.Tensor,
        statements_per_label: torch.Tensor,
        mask_value: float = -1e9) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut statements embeddings into flow statements embeddings

        Args:
            statements_embeddings (Tensor): [total n_statements; units]
            statements_per_label (Tensor): [n_flow]
            mask_value (float): -inf

        Returns: [n_flow; max flow n_statements; units], [n_flow; max flow n_statements]
        """
    batch_size = len(statements_per_label)
    max_context_len = max(statements_per_label)

    flow_statments_embeddings = statements_embeddings.new_zeros(
        (batch_size, max_context_len, statements_embeddings.shape[-1]))
    flow_statments_attn_mask = statements_embeddings.new_zeros(
        (batch_size, max_context_len))

    statments_slices = segment_sizes_to_slices(statements_per_label)
    for i, (cur_slice,
            cur_size) in enumerate(zip(statments_slices,
                                       statements_per_label)):
        flow_statments_embeddings[
            i, :cur_size] = statements_embeddings[cur_slice]
        flow_statments_attn_mask[i, cur_size:] = mask_value

    return flow_statments_embeddings, flow_statments_attn_mask


def gine_conv_nn(in_dim: int, out_dim: int) -> nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 2 * in_dim),
        torch.nn.BatchNorm1d(2 * in_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * in_dim, out_dim),
    )


class GINEConvEncoder(torch.nn.Module):
    """GINEConv encoder to encode each ast (statement) into a compact vector.
    We use GINEConv and TopKPooling based on JK-net-style architecture
    """
    def __init__(self, config: DictConfig, vocabulary_size: int, pad_idx: int):
        super().__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = nn.Embedding(vocabulary_size,
                                           config.ast.embed_dim,
                                           padding_idx=pad_idx)
        # Additional embedding value for masked token
        self.__node_type_embedding = nn.Embedding(
            len(NodeType) + 1, config.ast.embed_dim)
        torch.nn.init.xavier_uniform_(self.__st_embedding.weight.data)

        self.__input_GCL = GINEConv(
            gine_conv_nn(config.ast.embed_dim, config.ast.hidden_dim))
        self.__input_GPL = TopKPooling(config.ast.hidden_dim,
                                       ratio=config.ast.pooling_ratio)

        for i in range(config.ast.n_hidden_layers - 1):
            setattr(
                self, f"__hidden_GCL{i}",
                GINEConv(
                    gine_conv_nn(config.ast.hidden_dim,
                                 config.ast.hidden_dim)))
            setattr(
                self, f"__hidden_GPL{i}",
                TopKPooling(config.ast.hidden_dim,
                            ratio=config.ast.pooling_ratio))
        self.__fcl_after_cat = nn.Sequential(
            nn.Linear(2 * config.ast.hidden_dim, config.ast.hidden_dim),
            nn.Dropout(config.ast.dropout))

    def forward(self, batched_graph: Batch) -> torch.Tensor:
        """

        Args:
            batched_graph (Batch): [total n_statements (Data)]
            statements_per_label (Tensor): [n_flow]

        Returns: statement_embeddings: [total n_statements; ast hidden dim]
        """
        # [n nodes]
        n_parts = (batched_graph.x != self.__pad_idx).sum(dim=-1).reshape(
            -1, 1)
        # There are some nodes without token, e.g., `s = ""` would lead to node for "" with empty token.
        not_empty_mask = (n_parts != 0).reshape(-1)
        # [n nodes; embed dim]
        subtokens_embed = self.__st_embedding(batched_graph.x).sum(dim=1)
        subtokens_embed[not_empty_mask] /= n_parts[not_empty_mask]

        # [n nodes; embed dim]
        node_types_embed = self.__node_type_embedding(
            batched_graph["node_type"])
        # [n nodes; embed dim]
        node_embedding = subtokens_embed + node_types_embed

        # Sparse adjacent matrix
        # num_nodes = batched_graph.num_nodes
        # adj_t = SparseTensor.from_edge_index(batched_graph.edge_index, edge_embedding, (num_nodes, num_nodes)).t()
        # adj_t = adj_t.device_as(edge_embedding)

        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        # [n nodes; hidden dim]
        x = self.__input_GCL(x=node_embedding, edge_index=edge_index)
        x, edge_index, _, batch, _, _ = self.__input_GPL(x=x,
                                                         edge_index=edge_index,
                                                         edge_attr=None,
                                                         batch=batch)
        statement_embeddings = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        for i in range(self.__config.ast.n_hidden_layers - 1):
            x = getattr(self, f"__hidden_GCL{i}")(x, edge_index)
            x, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                x, edge_index, None, batch)
            statement_embeddings += torch.cat(
                [gmp(x, batch), gap(x, batch)], dim=1)

        # [total n_statements; ast hidden dim]
        statement_embeddings = self.__fcl_after_cat(statement_embeddings)
        return statement_embeddings


class FlowGRULayer(nn.Module):
    r"""GRU Layer for aggregate statements into flows
    """

    __negative_value = -numpy.inf

    def __init__(self, input_dim: int, out_dim: int, num_layers: int,
                 use_bi: bool, dropout: str, activation: str):
        self.__flow_gru = nn.GRU(input_size=input_dim,
                                 hidden_size=out_dim,
                                 num_layers=num_layers,
                                 bidirectional=use_bi,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.__dropout_flow_gru = nn.Dropout(dropout)
        self.__flow_att = LocalAttention(out_dim)
        self.__flow_hidden = linear_after_attn(out_dim, out_dim, activation)

    def forward(self, statements_embeddings: torch.Tensor,
                statements_per_label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            statements_embeddings (Tensor): [total n_statements, st hidden size]
            statements_per_label (Tensor): [n_flow]

        Returns: flow_embedding: [n_flow; flow_hidden_size]
        """
        # [n_flow; max flow n_statements; st hidden size], [n_flow; max flow n_statements]
        flow_statments_embeddings, flow_statments_attn_mask = cut_statements_embeddings(
            statements_embeddings, statements_per_label, self.__negative_value)

        with torch.no_grad():
            sorted_path_lengths, sort_indices = torch.sort(
                statements_per_label, descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        flow_statments_embeddings = flow_statments_embeddings[sort_indices]
        flow_statments_embeddings = nn.utils.rnn.pack_padded_sequence(
            flow_statments_embeddings, sorted_path_lengths, batch_first=True)
        flow_hiddens, _ = self.__flow_gru(flow_statments_embeddings)
        # [n_flow; max flow n_statements; 2*flow hidden size]
        flow_hiddens, _ = nn.utils.rnn.pad_packed_sequence(flow_hiddens,
                                                           batch_first=True)
        # [n_flow; max flow n_statements; 2*flow hidden size]
        flow_hiddens = self.__dropout_flow_gru(
            flow_hiddens)[reverse_sort_indices]
        # [n_flow; max flow n_statements; 1]
        flow_attn_weights = self.__flow_att(flow_hiddens,
                                            flow_statments_attn_mask)
        # [n_flow, flow hidden size]
        flow_embeddings = self.__flow_hidden(
            torch.bmm(flow_attn_weights.transpose(1, 2),
                      flow_hiddens).squeeze(1))
        return flow_embeddings, flow_attn_weights


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
        torch.nn.init.xavier_normal(self.__st_embedding.weight.data)

        self.__st_blstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.st_hidden_size,
            num_layers=config.st_num_layers,
            bidirectional=config.st_use_bi_rnn,
            dropout=config.st_dropout if config.st_num_layers > 1 else 0,
        )
        self.__dropout_st_blstm = nn.Dropout(config.st_dropout)
        self.__st_att = LocalAttention(config.st_hidden_size)
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
        st_hiddens, _ = nn.utils.rnn.pad_packed_sequence(st_hiddens)
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
        self.__flow_statments_weights, _ = cut_statements_embeddings(
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
    __activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "lkrelu": nn.LeakyReLU(0.3)
    }

    def __init__(self, config: DictConfig, pad_idx: int):
        super().__init__()
        self.__pad_idx = pad_idx
        model_name = "microsoft/codebert-base"
        bert_config = RobertaConfig.from_pretrained(model_name)
        self.__st_encoder = RobertaModel.from_pretrained(model_name,
                                                         config=bert_config)
        self.__flow_gru = FlowGRULayer(input_dim=bert_config.hidden_size,
                                       out_dim=config.flow_hidden_size,
                                       num_layers=config.num_layers,
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
        # self.__flow_statments_weights, _ = cut_statements_embeddings(
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
    def __init__(self, config: DictConfig, vocabulary_size: int, pad_idx: int):
        super().__init__()
        self.__gnn_encoder = GINEConvEncoder(config, vocabulary_size, pad_idx)
        self.__flow_gru = FlowGRULayer(input_dim=config.ast.hidden_dim,
                                       out_dim=config.flow_hidden_size,
                                       num_layers=config.num_layers,
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
        statements_embeddings = self.__gnn_encoder(batch)
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
    def __init__(self, config: DictConfig, vocabulary_size: int, pad_idx: int):
        super().__init__()
        # we can use the tokenizer of bert
        self.__lstm_encoder = FlowLSTMEncoder(config, vocabulary_size, pad_idx)
        self.__bert_encoder = FlowLSTMEncoder(config, vocabulary_size, pad_idx)
        self.__gnn_encoder = FlowGNNEncoder(config, vocabulary_size, pad_idx)
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
