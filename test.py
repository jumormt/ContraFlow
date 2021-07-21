from src.joern.ast_generator import build_ln_to_statement
from src.datas.datastructures import ASTNode, ASTGraph

if __name__ == "__main__":
    file_path = "/home/chengxiao/project/vul_detect/joern_slicer/main.cpp"
    nodes_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/nodes.csv"
    edges_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/edges.csv"
    ln_to_statements = build_ln_to_statement(file_path, nodes_path, edges_path)
    

    print()