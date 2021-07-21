import csv
from src.datas.datastructures import Statement, ASTNode, NodeType
from typing import Dict

file_path = "/home/chengxiao/project/vul_detect/joern_slicer/main.cpp"
nodes_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/nodes.csv"
edges_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/edges.csv"


def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


nodes = read_csv(nodes_path)
edges = read_csv(edges_path)

type_set = set()
node_info = dict()
ln_to_nodeid = dict()
for node_index, node in enumerate(nodes):
    assert isinstance(node, dict)
    if "key" in node.keys():
        node_info[int(node["key"])] = node
    if "type" in node and node["type"] == "CompoundStatement":
        continue
    if "location" in node.keys():
        location = node["location"]
        if location.strip() != '':
            try:
                ln = int(location.split(':')[0])
                if ln in ln_to_nodeid:
                    ln_to_nodeid[ln].append(int(node["key"]))
                else:
                    ln_to_nodeid[ln] = [int(node["key"])]
            except:
                pass
for edge_index, edge in enumerate(edges):
    if (edge["type"] == "IS_AST_PARENT"):
        if edge["start"] in node_info:
            if "childs" in node_info[edge["start"]]:
                node_info[int(edge["start"])]["childs"].append(int(
                    edge["end"]))
            else:
                node_info[int(edge["start"])]["childs"] = [int(edge["end"])]


def construct_ast(node_info: Dict, nodeid: str):
    if node_info[nodeid]["code"] != "" and node_info[nodeid]["code"][0] == '"':
        node_info[nodeid]["code"] = node_info[nodeid]["code"][1:-1]
    astnode = ASTNode(content=node_info[nodeid]["code"],
                      node_type=NodeType[node_info[nodeid]["type"]],
                      childs=list())

    if "childs" not in node_info[nodeid]:
        return astnode
    for child_nodeid in node_info[nodeid]["childs"]:
        astnode.childs.append(construct_ast(node_info, child_nodeid))
    return astnode


with open(file_path, encoding="utf-8") as f:
    file_content = f.readlines()

ln_to_statement = dict()
for ln in ln_to_nodeid:

    type_root = node_info[ln_to_nodeid[ln][0]]["type"]
    if type_root in ["Statement", "Function", "Condition"]:
        if type_root == "Condition":
            content = node_info[ln_to_nodeid[ln][0] - 1]["code"]
            type_ = node_info[ln_to_nodeid[ln][0] - 1]["type"]
        elif type_root == "Function":
            content = node_info[ln_to_nodeid[ln][0] + 1]["code"]
            type_ = node_info[ln_to_nodeid[ln][0] + 1]["type"]
        else:
            content = file_content[ln - 1].strip()
            type_ = type_root
        if (content != "" and content[0] == '"'):
            content = content[1:-1]
        astnode = ASTNode(content=content,
                          node_type=NodeType[type_],
                          childs=list())
        for nodeid in ln_to_nodeid[ln]:
            astnode.childs.append(construct_ast(node_info, nodeid))
    else:
        astnode = construct_ast(node_info, ln_to_nodeid[ln][0])

    statement = Statement(file_path=file_path, line_number=ln, ast=astnode)
    ln_to_statement[ln] = statement

print()