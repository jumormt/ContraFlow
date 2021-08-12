from src.joern.ast_generator import build_ln_to_ast
from transformers import RobertaTokenizer
from src.datas.graph import NodeType
from src.sequence_analyzer.sequence_analyzer import SequencesAnalyzer
import os
from os.path import join, exists
import json


def test_ast():
    file_path = "./data/joern/main.cpp"
    nodes_path = "./data/joern/nodes.csv"
    edges_path = "./data/joern/edges.csv"
    ln_to_ast = build_ln_to_ast(file_path, nodes_path, edges_path)
    ast_graph = ln_to_ast[39]
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    data = ast_graph.to_torch(tokenizer, 32)


def test_seq_sim():
    return SequencesAnalyzer("", "ab").similarity()


def test_ast_batch():
    if not exists("done.txt"):
        os.system("touch done.txt")
    with open("done.txt", "r") as f:
        done = set(f.read().split(","))
    data_root = "/home/chengxiao/project/ICSEDataSets/data/outputs/nginx"
    for commitid in os.listdir(data_root):
        if commitid in done:
            continue
        commitroot = join(data_root, commitid)

        file_json_path = join(commitroot, "files.json")
        fileroot = join(commitroot, "files")
        with open(file_json_path, "r") as f:
            file_json = json.load(f)
        for fil in file_json:
            graphroot = join(join(commitroot, "graphs"), fil)
            if (exists(graphroot)):
                nodes_path = join(graphroot, "nodes.csv")
                edges_path = join(graphroot, "edges.csv")
                file_path = join(fileroot, fil)
                print(nodes_path)
                print(file_path)
                ln_to_ast = build_ln_to_ast(file_path, nodes_path, edges_path)

        with open("done.txt", "a") as f:
            f.write(commitid + ",")


if __name__ == "__main__":
    # print(NodeType.__members__)
    print(test_seq_sim())
    # test_ast_batch()
