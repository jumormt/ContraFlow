from torch import nn
from omegaconf import DictConfig
import torch
from src.models.modules.attention import LocalAttention
import numpy
from src.utils import cut_lower_embeddings
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, TopKPooling, SuperGATConv
from src.datas.graph import NodeType
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


def gine_conv_nn(in_dim: int, out_dim: int) -> nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * in_dim),
                               torch.nn.BatchNorm1d(2 * in_dim),
                               torch.nn.ReLU(),
                               torch.nn.Linear(2 * in_dim, out_dim))


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


class SuperGATConvEncoder(torch.nn.Module):
    """SUPERGATConv encoder to encode each ast (statement) into a compact vector.

    We use SuperGATConv from  "How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision” (ICLR'21). The architecture is based on JK-net-style architecture.

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

        self.__input_GCL = SuperGATConv(
            in_channels=config.ast.embed_dim,
            out_channels=config.ast.hidden_dim,
            heads=config.ast.n_head,
            dropout=config.ast.dropout,
            attention_type='MX',
            edge_sample_ratio=config.ast.edge_sample_ratio)

        for i in range(config.ast.n_hidden_layers - 1):
            setattr(
                self, f"__hidden_GCL{i}",
                SuperGATConv(in_channels=config.ast.n_head *
                             config.ast.hidden_dim,
                             out_channels=config.ast.hidden_dim,
                             heads=config.ast.n_head,
                             dropout=config.ast.dropout,
                             attention_type='MX',
                             edge_sample_ratio=config.ast.edge_sample_ratio))
        self.__fcl_after_cat = nn.Sequential(
            nn.Linear(2 * config.ast.n_head * config.ast.hidden_dim,
                      config.ast.hidden_dim), nn.Dropout(config.ast.dropout))

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
        # [n nodes; n_head * hidden dim]
        x = self.__input_GCL(x=node_embedding, edge_index=edge_index)
        self.__att_loss = self.__input_GCL.get_attention_loss()
        statement_embeddings = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        for i in range(self.__config.ast.n_hidden_layers - 1):
            x = getattr(self, f"__hidden_GCL{i}")(x, edge_index)
            statement_embeddings += torch.cat(
                [gmp(x, batch), gap(x, batch)], dim=1)
            self.__att_loss += getattr(
                self, f"__hidden_GCL{i}").get_attention_loss()

        # [total n_statements; ast hidden dim]
        statement_embeddings = self.__fcl_after_cat(statement_embeddings)
        return statement_embeddings

    def get_att_loss(self):
        return self.__att_loss


class FlowGRULayer(nn.Module):
    r"""GRU Layer for aggregate statements into flows
    """

    __negative_value = -numpy.inf

    def __init__(self, input_dim: int, out_dim: int, num_layers: int,
                 use_bi: bool, dropout: str, activation: str):
        super().__init__()
        self.__flow_gru = nn.GRU(input_size=input_dim,
                                 hidden_size=out_dim,
                                 num_layers=num_layers,
                                 bidirectional=use_bi,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.__dropout_flow_gru = nn.Dropout(dropout)
        self.__flow_att = LocalAttention(2 * out_dim)
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
        flow_statments_embeddings, flow_statments_attn_mask = cut_lower_embeddings(
            statements_embeddings, statements_per_label, self.__negative_value)
        max_flow_n_statements = flow_statments_embeddings.size(1)

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
        flow_hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            flow_hiddens,
            batch_first=True,
            total_length=max_flow_n_statements)
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
