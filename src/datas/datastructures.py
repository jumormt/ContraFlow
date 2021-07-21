from dataclasses import dataclass
from typing import Set, Dict, List
from torch_geometric.data import Data

import numpy
import torch
from enum import Enum
from pyvis.network import Network
from src.utils import traverse_ast
from transformers import RobertaTokenizer


@dataclass
class ValueFlow:
    statements: numpy.ndarray  # [seq len; n_statements]
    n_statements: int
    statements_idx: List[int] = None
    feature: numpy.ndarray = None  # [feature dim,]


class ValueFlowBatch:
    def __init__(self, value_flows: List[ValueFlow]):
        # [batch size]
        self.statements_per_label = torch.tensor(
            [value_flow.n_statements for value_flow in value_flows],
            dtype=torch.long)

        # [seq len; total n_statements]
        self.statements = torch.from_numpy(
            numpy.hstack([value_flow.statements
                          for value_flow in value_flows]))

        # [batch size, feature dim]
        self.features = numpy.full(
            (len(value_flows), value_flows[0].feature.shape[0]),
            0,
            dtype=numpy.long)

        for i, value_flow in enumerate(value_flows):
            self.features[i] = value_flow.feature
        self.features = torch.from_numpy(self.features)

    def __len__(self):
        return len(self.statements_per_label)

    def pin_memory(self) -> "ValueFlowBatch":
        self.statements_per_label = self.statements_per_label.pin_memory()
        self.statements = self.statements.pin_memory()
        self.features = self.features.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.statements_per_label = self.statements_per_label.to(device)
        self.statements = self.statements.to(device)
        self.features = self.features.to(device)


@dataclass
class ValueFlowPair:
    value_flow_1: ValueFlow
    value_flow_2: ValueFlow


class ValueFlowPairBatch:
    def __init__(self, value_flow_pairs: List[ValueFlowPair]):
        # [batch size]
        self.statements_per_label1 = torch.tensor(
            [_pair.value_flow_1.n_statements for _pair in value_flow_pairs],
            dtype=torch.long)
        self.statements_per_label2 = torch.tensor(
            [_pair.value_flow_2.n_statements for _pair in value_flow_pairs],
            dtype=torch.long)

        # [seq len; total n_statements]
        self.statements1 = torch.from_numpy(
            numpy.hstack(
                [_pair.value_flow_1.statements for _pair in value_flow_pairs]))
        self.statements2 = torch.from_numpy(
            numpy.hstack(
                [_pair.value_flow_2.statements for _pair in value_flow_pairs]))

    def __len__(self):
        return len(self.statements_per_label1)

    def pin_memory(self) -> "ValueFlowPairBatch":
        self.statements_per_label1 = self.statements_per_label1.pin_memory()
        self.statements_per_label2 = self.statements_per_label2.pin_memory()
        self.statements1 = self.statements1.pin_memory()
        self.statements2 = self.statements2.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.statements_per_label1 = self.statements_per_label1.to(device)
        self.statements_per_label2 = self.statements_per_label1.to(device)
        self.statements1 = self.statements1.to(device)
        self.statements2 = self.statements2.to(device)


@dataclass
class MethodSample:
    value_flows: List[ValueFlow]
    label: int
    flaws: List[int]


class MethodSampleBatch:
    def __init__(self, method_samples: List[MethodSample]):
        # [batch size]
        self.value_flow_per_label = torch.tensor([
            len(method_sample.value_flows) for method_sample in method_samples
        ],
                                                 dtype=torch.long)
        self.labels = torch.tensor(
            [method_sample.label for method_sample in method_samples],
            dtype=torch.long)
        self.flaws: List[List[int]] = [
            method_sample.flaws for method_sample in method_samples
        ]

        self.statements_per_value_flow = list()
        self.statements = list()
        self.statements_idxes = list()

        for method_sample in method_samples:
            self.statements_per_value_flow.extend([
                value_flow.n_statements
                for value_flow in method_sample.value_flows
            ])
            self.statements.append(
                # [seq len; method n_statements]
                torch.from_numpy(
                    numpy.hstack([
                        value_flow.statements
                        for value_flow in method_sample.value_flows
                    ])))
            self.statements_idxes.append([
                value_flow.statements_idx
                for value_flow in method_sample.value_flows
            ])
        # [seq len; total n_statements]
        self.statements = torch.cat(self.statements, dim=1)
        # [total n_flow]
        self.statements_per_value_flow = torch.tensor(
            self.statements_per_value_flow, dtype=torch.long)

    def __len__(self):
        return len(self.value_flow_per_label)

    def pin_memory(self) -> "MethodSampleBatch":
        self.value_flow_per_label = self.value_flow_per_label.pin_memory()
        self.statements_per_value_flow = self.statements_per_value_flow.pin_memory(
        )
        self.statements = self.statements.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.value_flow_per_label = self.value_flow_per_label.to(device)
        self.statements_per_value_flow = self.statements_per_value_flow.to(
            device)
        self.statements = self.statements.to(device)
        self.labels = self.labels.to(device)


class NodeType(Enum):
    """Enum class to represent node type.
    """

    CompoundStatement = 0
    ExpressionStatement = 1
    IdentifierDeclStatement = 2
    IfStatement = 3
    ReturnStatement = 4
    WhileStatement = 5
    ElseStatement = 6
    BreakStatement = 7
    Statement = 8

    FunctionDef = 9
    IdentifierDecl = 10
    AdditiveExpression = 11
    IdentifierDeclType = 12
    MemberAccess = 13
    CFGEntryNode = 14
    File = 15
    Symbol = 16
    ArrayIndexing = 17
    Parameter = 18
    SizeofExpression = 19
    AssignmentExpression = 20
    ParameterType = 21
    Condition = 22
    EqualityExpression = 23
    ParameterList = 24
    Decl = 25
    Callee = 26
    Argument = 27
    ReturnType = 28
    ArgumentList = 29
    SizeofOperand = 30
    UnaryOperationExpression = 31
    ShiftExpression = 32
    PtrMemberAccess = 33
    DeclStmt = 34
    OrExpression = 35
    Sizeof = 36
    Function = 37
    UnaryOperator = 38
    Identifier = 39
    CFGExitNode = 40
    PrimaryExpression = 41
    RelationalExpression = 42
    CallExpression = 43


@dataclass
class ASTNode:
    content: str
    node_type: NodeType
    childs: List["ASTNode"]


@dataclass
class ASTEdge:
    from_node: ASTNode
    to_node: ASTNode


@dataclass
class Statement:
    file_path: str
    line_number: int
    ast: ASTNode


class ASTGraph:
    def __init__(self, nodes: List[ASTNode], edges: List[ASTEdge]):
        self.__nodes = nodes
        self.__edges = edges

    @staticmethod
    def from_root_ast(ast: ASTNode) -> "ASTGraph":
        nodes, edges = zip(*list(traverse_ast(ast)))
        nodes, edges = list(nodes), list(edges)
        nodes.append(ast)
        return ASTGraph(nodes, edges)

    @property
    def nodes(self) -> List[ASTNode]:
        return self.__nodes

    @property
    def edges(self) -> List[ASTEdge]:
        return self.__edges

    def to_torch(self, tokenizer: RobertaTokenizer, max_len: int) -> Data:
        """Convert this graph into torch-geometric graph

        Args:
            tokenizer: tokenizer to convert token parts into ids
            max_len: vector max_len for node content
        Returns:
            :torch_geometric.data.Data
        """
        node_tokens = [tokenizer.tokenize(n.content) for n in self.nodes]
        # [n_node, max seq len]
        node_ids = torch.full((len(node_tokens), max_len),
                              tokenizer.pad_token_id,
                              dtype=torch.long)
        for tokens_idx, tokens in enumerate(node_tokens):
            ids = tokenizer.convert_tokens_to_ids(tokens)
            less_len = min(max_len, len(ids))
            node_ids[tokens_idx, :less_len] = torch.tensor(ids[:less_len],
                                                           dtype=torch.long)

        # [n_node]
        node_type = torch.tensor([n.node_type.value for n in self.nodes],
                                 dtype=torch.long)
        edge_index = torch.tensor(list(
            zip(*[[self.node.index(e.from_node),
                   self.node.index(e.to_node)] for e in self.edges])),
                                  dtype=torch.long)

        # save token to `x` so Data can calculate properties like `num_nodes`
        return Data(x=node_ids, node_type=node_type, edge_index=edge_index)

    def draw(self,
             height: int = 1000,
             width: int = 1000,
             notebook: bool = True) -> Network:
        """Visualize graph using [pyvis](https://pyvis.readthedocs.io/en/latest/) library

        :param graph: graph instance to visualize
        :param height: height of target visualization
        :param width: width of target visualization
        :param notebook: pass True if visualization should be displayed in notebook
        :return: pyvis Network instance
        """
        net = Network(height=height,
                      width=width,
                      directed=True,
                      notebook=notebook)
        net.barnes_hut(gravity=-10000, overlap=1, spring_length=1)

        for node in self.nodes:
            net.add_node(node.id,
                         label=node.token,
                         group=node.type.value,
                         title=f"id: {node.id}\ntoken: {node.token}")

        for edge in self.edges:
            net.add_edge(edge.from_node.id,
                         edge.to_node.id,
                         label=edge.type.name,
                         group=edge.type.value)

        return net
