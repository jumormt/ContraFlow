from enum import Enum
from pyvis.network import Network
from transformers import RobertaTokenizer
from typing import List
import torch
from torch_geometric.data import Data
from dataclasses import dataclass


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
    CastExpression = 44
    CastTarget = 45

    PLAIN = 46

    PostIncDecOperationExpression = 47
    IncDec = 48
    UnaryExpression = 49
    AndExpression = 50
    ConditionalExpression = 51
    MultiplicativeExpression = 52
    SwitchStatement = 53
    Label = 54
    ContinueStatement = 55
    ForInit = 56
    ForStatement = 57
    DoStatement = 58
    BitAndExpression = 59
    InclusiveOrExpression = 60
    InitializerList = 61
    ClassDefStatement = 62
    GotoStatement = 63
    ClassDef = 64
    Expression = 65
    ExclusiveOrExpression = 66




@dataclass
class ASTNode:
    content: str
    node_type: NodeType
    childs: List["ASTNode"]


@dataclass
class ASTEdge:
    from_node: ASTNode
    to_node: ASTNode


class ASTGraph:
    def __init__(self, nodes: List[ASTNode], edges: List[ASTEdge]):
        self.__nodes = nodes
        self.__edges = edges

    @staticmethod
    def from_root_ast(ast: ASTNode) -> "ASTGraph":
        if len(ast.childs) == 0:
            nodes = []
            edges = []
        else:
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
            zip(*[[self.nodes.index(e.from_node),
                   self.nodes.index(e.to_node)] for e in self.edges])),
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

        for idx, node in enumerate(self.nodes):
            net.add_node(
                idx,
                label=node.content,
                group=node.node_type.value,
                title=f"type:{node.node_type.name}\ntoken: {node.content}")

        for edge in self.edges:
            net.add_edge(self.nodes.index(edge.from_node),
                         self.nodes.index(edge.to_node),
                         label=None,
                         group=None)

        return net


def traverse_ast(ast: ASTNode):
    for child in ast.childs:
        yield (child, ASTEdge(from_node=child, to_node=ast))
        yield from traverse_ast(child)
