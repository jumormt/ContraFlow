from src.joern.ast_generator import build_ln_to_ast
from transformers import RobertaTokenizer

if __name__ == "__main__":
    file_path = "/home/chengxiao/project/vul_detect/joern_slicer/main.cpp"
    nodes_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/nodes.csv"
    edges_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/edges.csv"
    ln_to_ast = build_ln_to_ast(file_path, nodes_path, edges_path)
    ast_graph = ln_to_ast[39]
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    data = ast_graph.to_torch(tokenizer, 32)

    print()