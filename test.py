from src.joern.ast_generator import build_ln_to_ast
from transformers import RobertaTokenizer
from src.datas.graph import NodeType


def test_ast():
    file_path = "./data/joern/main.cpp"
    nodes_path = "./data/joern/nodes.csv"
    edges_path = "./data/joern/edges.csv"
    ln_to_ast = build_ln_to_ast(file_path, nodes_path, edges_path)
    ast_graph = ln_to_ast[39]
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    data = ast_graph.to_torch(tokenizer, 32)


if __name__ == "__main__":
    print(NodeType.__members__)
