from src.datas.graph import ASTGraph, ASTNode, NodeType
from typing import Dict, List, Tuple
from src.utils import read_csv


def build_node_metadata(
        nodes: List[Dict]) -> Tuple[Dict[int, Dict], Dict[int, List[int]]]:
    """
    Returns:
        node_info: {nodekey: infodict}
        ln_to_nodeid: {line: nodekey}
    """
    node_info = dict()
    ln_to_nodeid = dict()
    for node in nodes:
        assert isinstance(node, dict)
        if "key" in node.keys():
            node_info[int(node["key"])] = node
        if "type" in node and node["type"] == "CompoundStatement":
            continue
        if "location" in node.keys():
            location = node["location"]
            if location.strip() != '' and len(location.split(':')) != 0:

                ln = int(location.split(':')[0])
                if ln in ln_to_nodeid:
                    ln_to_nodeid[ln].append(int(node["key"]))
                else:
                    ln_to_nodeid[ln] = [int(node["key"])]

    return node_info, ln_to_nodeid


def add_ast_childs(edges: List[Dict],
                   node_info: Dict[int, Dict]) -> Dict[int, Dict]:
    """

    add "childs" to statement node info

    Returns:
        node_info: {nodekey: infodict}
    """
    for edge in edges:
        if ("type" in edge and edge["type"] == "IS_AST_PARENT"):
            assert "start" in edge and "end" in edge, f"{edge} do not contain key 'start' 'edge'"
            if int(edge["start"]) in node_info:
                if "childs" in node_info[int(edge["start"])]:
                    node_info[int(edge["start"])]["childs"].append(
                        int(edge["end"]))
                else:
                    node_info[int(
                        edge["start"])]["childs"] = [int(edge["end"])]

    return node_info


def construct_ast(node_info: Dict, nodeid: int) -> ASTNode:
    """
    recursively construct ast
    """
    assert nodeid in node_info, f"{nodeid} not in node_info"
    assert "code" in node_info[nodeid], f"'code' not in {node_info[nodeid]}"

    if node_info[nodeid]["code"] != "" and node_info[nodeid]["code"][0] == '"':
        node_info[nodeid]["code"] = node_info[nodeid]["code"][1:-1]
    assert "type" in node_info[nodeid], f"'type' not in {node_info[nodeid]}"
    assert node_info[nodeid][
        "type"] in NodeType.__members__, f"unknown node type {node_info[nodeid]['type']}"
    astnode = ASTNode(content=node_info[nodeid]["code"],
                      node_type=NodeType[node_info[nodeid]["type"]],
                      childs=list())

    if "childs" not in node_info[nodeid]:
        return astnode
    for child_nodeid in node_info[nodeid]["childs"]:
        astnode.childs.append(construct_ast(node_info, child_nodeid))
    return astnode


def build_ln_to_ast(file_path: str, nodes_path: str,
                    edges_path: str) -> Dict[int, ASTGraph]:
    """
    
    Args:
        file_path (str): 
        nodes_path (str): node csv file
        edges_path (str): edge csv file
    
    Returns:
        ln_to_statement: line to Statement
    """
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    node_info, ln_to_nodeid = build_node_metadata(nodes)
    node_info = add_ast_childs(edges, node_info)
    with open(file_path, encoding="utf-8") as f:
        file_content = f.readlines()

    ln_to_ast_graph = dict()
    for ln in ln_to_nodeid:

        type_root = node_info[ln_to_nodeid[ln][0]]["type"]
        if type_root in ["Statement", "Function", "Condition"]:
            if type_root == "Condition":
                assert ln_to_nodeid[ln][
                    0] - 1 in node_info, f"{ln_to_nodeid[ln][0] - 1} not in node_info"
                assert "code" in node_info[
                    ln_to_nodeid[ln][0] -
                    1], f"'code' not in {node_info[ln_to_nodeid[ln][0] - 1]}"
                content = node_info[ln_to_nodeid[ln][0] - 1]["code"]
                type_ = node_info[ln_to_nodeid[ln][0] - 1]["type"]
            elif type_root == "Function":
                assert ln_to_nodeid[ln][
                    0] + 1 in node_info, f"{ln_to_nodeid[ln][0] + 1} not in node_info"
                assert "code" in node_info[
                    ln_to_nodeid[ln][0] -
                    1], f"'code' not in {node_info[ln_to_nodeid[ln][0] - 1]}"
                content = node_info[ln_to_nodeid[ln][0] + 1]["code"]
                type_ = node_info[ln_to_nodeid[ln][0] + 1]["type"]
            else:
                assert ln - 1 < len(
                    file_content
                ), f"{file_path} and {nodes_path} does not match"
                content = file_content[ln - 1].strip()
                type_ = type_root
            if (content != "" and content[0] == '"'):
                content = content[1:-1]
            assert type_ in NodeType.__members__, f"unkown node type: {type_}"
            astnode = ASTNode(content=content,
                              node_type=NodeType[type_],
                              childs=list())
            for nodeid in ln_to_nodeid[ln]:
                astnode.childs.append(construct_ast(node_info, nodeid))
        else:
            astnode = construct_ast(node_info, ln_to_nodeid[ln][0])

        ast_graph = ASTGraph.from_root_ast(astnode)
        ln_to_ast_graph[ln] = ast_graph
    return ln_to_ast_graph


if __name__ == "__main__":
    file_path = "/home/chengxiao/project/vul_detect/joern_slicer/main.cpp"
    nodes_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/nodes.csv"
    edges_path = "/home/chengxiao/project/vul_detect/joern_slicer/output/main.cpp/edges.csv"
    ln_to_statements = build_ln_to_ast(file_path, nodes_path, edges_path)
