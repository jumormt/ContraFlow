import numpy
from torch.utils.data import Dataset
import json
from tokenizers import Tokenizer
from omegaconf import DictConfig
from src.datas.graph import ASTNode, ASTGraph, NodeType
from src.datas.samples import ValueFlowPair, ValueFlow, MethodSample
from src.utils import strings_to_numpy, PAD, get_ast_path_from_file
from os.path import exists
from transformers import RobertaTokenizer
from typing import Union
from src.joern.ast_generator import build_ln_to_ast
from torch_geometric.data import Data
from os.path import join


class ASTDataset(Dataset):
    """
    ["dataset/project/commitid/files/*"]
    """
    def __init__(self, data_path: str, config: DictConfig,
                 tokenizer: Union[Tokenizer, RobertaTokenizer]) -> None:
        super().__init__()
        self.__config = config
        if not exists(data_path):
            raise ValueError(f"Can't find file with data: {data_path}")
        with open(data_path, "r") as f:
            self.__files = list(json.load(f))
        self.__n_samples = 0
        self.__asts = list()
        for file_path in self.__files:
            nodes_path, edges_path = get_ast_path_from_file(file_path)
            ln_to_ast_graph = build_ln_to_ast(file_path, nodes_path,
                                              edges_path)
            self.__asts.extend([ln_to_ast_graph[ln] for ln in ln_to_ast_graph])
            self.__n_samples += len(ln_to_ast_graph)

        self.__tokenizer = tokenizer

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> Data:

        return self.__asts[index].to_torch(self.__tokenizer,
                                           self.__config.max_token_parts)

    def get_n_samples(self):
        return self.__n_samples


class ValueFlowDataset(Dataset):
    """
    [{"files": ["dataset/project/commitid/files/*"], "graph_paths": ["dataset/project/commitid/graphs/*"], "flow": ["0,1", "0,2"], "apis":"123", "types":"123"}]

    """
    def __init__(self, data_path: str, config: DictConfig,
                 tokenizer: Union[Tokenizer, RobertaTokenizer]) -> None:
        super().__init__()
        self.__config = config
        if not exists(data_path):
            raise ValueError(f"Can't find file with data: {data_path}")
        with open(data_path, "r") as f:
            self.__value_flows = list(json.load(f))
        self.__n_samples = len(self.__value_flows)
        self.__tokenizer = tokenizer
        if config.encoder.name == "LSTM":
            # self.__tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD),
            #                                 pad_token=PAD,
            #                                 length=config.max_token_parts)
            # self.__tokenizer.enable_truncation(
            #     max_length=config.max_token_parts)
            pass
        elif config.encoder.name == "BERT":
            pass
        elif config.encoder.name == "GNN":
            pass
        elif config.encoder.name == "HYBRID":
            pass
        else:
            raise ValueError(f"Can't find model: {self.encoder.name}")

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> ValueFlow:
        value_flow = self.__value_flows[index]
        assert "files" in value_flow, f"{value_flow} do not contain key 'file'"
        file_paths = value_flow["files"]
        ln_to_ast_graphs = list()
        for idx, graph_path in enumerate(value_flow["graph_paths"]):

            nodes_path, edges_path = join(graph_path, "nodes.csv"), join(
                graph_path, "edges.csv")
            ln_to_ast_graph = build_ln_to_ast(file_paths[idx], nodes_path,
                                              edges_path)
            ln_to_ast_graphs.append(ln_to_ast_graph)
        file_contents = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.readlines()
            file_contents.append(file_content)

        assert "flow" in value_flow, f"{value_flow} do not contain key 'flow'"
        value_flow_lines = value_flow["flow"]
        value_flow_raw = []
        ast_graphs = []
        for line in value_flow_lines:
            spts = line.split(",")
            file_idx, ln = int(spts[0]), int(spts[1])

            assert ln - 2 < len(
                file_contents[file_idx]
            ), f"value flow line overflow, check: f{file_paths[file_idx]}"
            if (ln - 1 == len(file_contents[file_idx])):
                ln -= 1
            line_raw = file_contents[file_idx][ln - 1].strip()
            value_flow_raw.append(line_raw)
            if ln in ln_to_ast_graphs[file_idx]:
                ast_graphs.append(ln_to_ast_graphs[file_idx][ln].to_torch(
                    self.__tokenizer, self.__config.max_token_parts))
            else:
                ast_graphs.append(
                    ASTGraph.from_root_ast(
                        ASTNode(content=line_raw,
                                node_type=NodeType.PLAIN,
                                childs=[])).to_torch(
                                    self.__tokenizer,
                                    self.__config.max_token_parts))
        # assert "feature" in value_flow, f"{value_flow} do not contain key 'feature'"
        # feature = value_flow["feature"]
        assert "apis" in value_flow, f"{value_flow} do not contain key 'apis'"
        assert "types" in value_flow, f"{value_flow} do not contain key 'types'"
        apis, types = value_flow["apis"], value_flow["types"]
        statements = strings_to_numpy(value_flow_raw, self.__tokenizer,
                                      self.__config.encoder.name,
                                      self.__config.max_token_parts)

        return ValueFlow(
            statements=statements,
            n_statements=len(value_flow_raw),
            #  feature=numpy.array(feature),
            sequence=(apis, types),
            ast_graphs=ast_graphs)

    def get_n_samples(self):
        return self.__n_samples


class ValueFlowPairDataset(Dataset):
    """
    [[{"file":, "flow_linenum": []},{"file":, "flow_linenum": []}],]

    """
    def __init__(self, data_path: str, config: DictConfig,
                 tokenizer: Union[Tokenizer, RobertaTokenizer]) -> None:
        super().__init__()
        self.__config = config
        if not exists(data_path):
            raise ValueError(f"Can't find file with data: {data_path}")
        with open(data_path, "r") as f:
            self.__pairs = list(json.load(f))
        self.__n_samples = len(self.__pairs)
        self.__tokenizer = tokenizer
        if config.encoder.name == "LSTM":
            self.__tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD),
                                            pad_token=PAD,
                                            length=config.max_token_parts)
            self.__tokenizer.enable_truncation(
                max_length=config.max_token_parts)
        elif config.encoder.name == "BERT":
            pass
        elif config.encoder.name == "GNN":
            pass
        elif config.encoder.name == "HYBRID":
            pass
        else:
            raise ValueError(f"Can't find model: {self.encoder.name}")

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> ValueFlowPair:
        pair = self.__pairs[index]
        assert len(pair) == 2, f"{pair} should be a value flow pair!"
        assert "file" in pair[0], f"{pair[0]} do not contain key 'file'"
        file_path1 = pair[0]["file"]
        nodes_path1, edges_path1 = get_ast_path_from_file(file_path1)
        with open(file_path1, "r", encoding="utf-8", errors="ignore") as f:
            file_content1 = f.readlines()
        ln_to_ast_graph1 = build_ln_to_ast(file_path1, nodes_path1,
                                           edges_path1)
        assert "flow" in pair[0], f"{pair[0]} do not contain key 'flow'"
        value_flow_lines1 = pair[0]["flow"]
        value_flow_raw1 = []
        ast_graphs1 = []
        for line in value_flow_lines1:
            line_raw = file_content1[line - 1].strip()
            value_flow_raw1.append(line_raw)
            if line in ln_to_ast_graph1:
                ast_graphs1.append(ln_to_ast_graph1[line].to_torch(
                    self.__tokenizer, self.__config.max_token_parts))
            else:
                ast_graphs1.append(
                    ASTGraph.from_root_ast(
                        ASTNode(content=line_raw,
                                node_type=NodeType.PLAIN,
                                childs=[])).to_torch(
                                    self.__tokenizer,
                                    self.__config.max_token_parts))
        statements1 = strings_to_numpy(value_flow_raw1, self.__tokenizer,
                                       self.__config.encoder.name,
                                       self.__config.max_token_parts)
        value_flow1 = ValueFlow(statements=statements1,
                                n_statements=len(value_flow_raw1))

        assert "file" in pair[1], f"{pair[1]} do not contain key 'file'"
        file_path2 = pair[1]["file"]
        nodes_path2, edges_path2 = get_ast_path_from_file(file_path2)
        with open(file_path2, "r", encoding="utf-8", errors="ignore") as f:
            file_content2 = f.readlines()
        ln_to_ast_graph2 = build_ln_to_ast(file_path2, nodes_path2,
                                           edges_path2)
        assert "flow" in pair[1], f"{pair[1]} do not contain key 'flow'"
        value_flow_lines2 = pair[1]["flow"]
        value_flow_raw2 = []
        ast_graphs2 = []
        for line in value_flow_lines2:
            line_raw = file_content2[line - 1].strip()
            value_flow_raw2.append(line_raw)
            if line in ln_to_ast_graph2:
                ast_graphs2.append(ln_to_ast_graph2[line].to_torch(
                    self.__tokenizer, self.__config.max_token_parts))
            else:
                ast_graphs2.append(
                    ASTGraph.from_root_ast(
                        ASTNode(content=line_raw,
                                node_type=NodeType.PLAIN,
                                childs=[])).to_torch(
                                    self.__tokenizer,
                                    self.__config.max_token_parts))
        statements2 = strings_to_numpy(value_flow_raw2, self.__tokenizer,
                                       self.__config.encoder.name,
                                       self.__config.max_token_parts)
        value_flow2 = ValueFlow(statements=statements2,
                                n_statements=len(value_flow_raw2))

        return ValueFlowPair(value_flow_1=value_flow1,
                             value_flow_2=value_flow2)

    def get_n_samples(self):
        return self.__n_samples


class MethodSampleDataset(Dataset):
    """
    [{"files": ["dataset/project/commitid/files/*"], "graph_paths": ["dataset/project/commitid/graphs/*"], "label":0, "flaws":["file idx,line",], "flows": [["file idx,line",], ...]}]
    """
    def __init__(self, data_path: str, config: DictConfig,
                 tokenizer: Union[Tokenizer, RobertaTokenizer]) -> None:
        super().__init__()
        self.__config = config
        if not exists(data_path):
            raise ValueError(f"Can't find file with data: {data_path}")
        with open(data_path, "r") as f:
            self.__methods = list(json.load(f))
        self.__n_samples = len(self.__methods)
        self.__tokenizer = tokenizer
        if config.encoder.name == "LSTM":
            # self.__tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD),
            #                                 pad_token=PAD,
            #                                 length=config.max_token_parts)
            # self.__tokenizer.enable_truncation(
            #     max_length=config.max_token_parts)
            pass
        elif config.encoder.name == "BERT":
            pass
        elif config.encoder.name == "GNN":
            pass
        elif config.encoder.name == "HYBRID":
            pass
        else:
            raise ValueError(f"Can't find model: {self.encoder.name}")

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> MethodSample:
        method = self.__methods[index]
        assert "flows" in method, f"{method} do not contain key 'flows'"
        n_flow = min(len(method["flows"]),
                     self.__config.hyper_parameters.max_n_flow)
        flow_indexes = numpy.arange(n_flow)
        numpy.random.shuffle(flow_indexes)

        value_flows = list()
        assert "files" in method, f"{method} do not contain key 'file'"
        file_paths = method["files"]
        ln_to_ast_graphs = list()
        for idx, graph_path in enumerate(method["graph_paths"]):

            nodes_path, edges_path = join(graph_path, "nodes.csv"), join(
                graph_path, "edges.csv")
            ln_to_ast_graph = build_ln_to_ast(file_paths[idx], nodes_path,
                                              edges_path)
            ln_to_ast_graphs.append(ln_to_ast_graph)
        file_contents = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.readlines()
            file_contents.append(file_content)

        for i in flow_indexes:
            value_flow_lines = method["flows"][i]
            value_flow_raw = []
            ast_graphs = []
            for line in value_flow_lines:
                spts = line.split(",")
                file_idx, ln = int(spts[0]), int(spts[1])
                assert ln - 2 < len(
                    file_contents[file_idx]
                ), f"value flow line overflow, check: f{file_paths[file_idx]}"
                if (ln - 1 == len(file_contents[file_idx])):
                    ln -= 1
                line_raw = file_contents[file_idx][ln - 1].strip()
                value_flow_raw.append(line_raw)
                if ln in ln_to_ast_graphs[file_idx]:
                    ast_graphs.append(ln_to_ast_graphs[file_idx][ln].to_torch(
                        self.__tokenizer, self.__config.max_token_parts))
                else:
                    ast_graphs.append(
                        ASTGraph.from_root_ast(
                            ASTNode(content=line_raw,
                                    node_type=NodeType.PLAIN,
                                    childs=[])).to_torch(
                                        self.__tokenizer,
                                        self.__config.max_token_parts))
            statements = strings_to_numpy(value_flow_raw, self.__tokenizer,
                                          self.__config.encoder.name,
                                          self.__config.max_token_parts)
            value_flows.append(
                ValueFlow(statements=statements,
                          n_statements=len(value_flow_raw),
                          statements_idx=value_flow_lines,
                          ast_graphs=ast_graphs))
        assert "label" in method, f"{method} do not contain key 'label'"
        assert "flaws" in method, f"{method} do not contain key 'flaws'"
        return MethodSample(value_flows=value_flows,
                            label=method["label"],
                            flaws=method["flaws"])

    def get_n_samples(self):
        return self.__n_samples
