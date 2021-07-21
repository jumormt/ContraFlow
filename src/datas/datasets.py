import numpy
from torch.utils.data import Dataset
import json
from tokenizers import Tokenizer
from omegaconf import DictConfig
from src.datas.datastructures import ValueFlowPair, ValueFlow, MethodSample
from src.utils import strings_to_numpy, PAD
from os.path import exists
from transformers import RobertaTokenizer
from typing import Union


class ValueFlowDataset(Dataset):
    """
    [{"file": "dataset/project/commitid/files/*", "flow": [line 1, line 2, ...], "feature":[1,0,1]}]

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
            self.__tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD),
                                            pad_token=PAD,
                                            length=config.max_token_parts)
            self.__tokenizer.enable_truncation(
                max_length=config.max_token_parts)
        elif config.encoder.name == "BERT":
            pass
        elif config.encoder.name == "HYBRID":
            pass
        else:
            raise ValueError(f"Can't find model: {self.encoder.name}")

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> ValueFlow:
        value_flow = self.__value_flows[index]
        with open(value_flow["file"], encoding="utf-8", errors="ignore") as f:
            file_content = f.readlines()
        value_flow_lines = value_flow["flow"]
        value_flow_raw = [file_content[line - 1] for line in value_flow_lines]
        feature = value_flow["feature"]
        statements = strings_to_numpy(value_flow_raw, self.__tokenizer,
                                      self.__config.encoder.name,
                                      self.__config.max_token_parts)

        return ValueFlow(statements=statements,
                         n_statements=len(value_flow_raw),
                         feature=numpy.array(feature))

    def get_n_samples(self):
        return self.__n_samples


class ValueFlowPairDataset(Dataset):
    """
    [[{"file":, "flow_linenum": [], "feature":[1,0,1]},{"file":, "flow_linenum": [], "feature":[1,0,1]}],]

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
        elif config.encoder.name == "HYBRID":
            pass
        else:
            raise ValueError(f"Can't find model: {self.encoder.name}")

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> ValueFlowPair:
        pair = self.__pairs[index]
        with open(pair[0]["file"], encoding="utf-8", errors="ignore") as f:
            file_content1 = f.readlines()
        with open(pair[1]["file"], encoding="utf-8", errors="ignore") as f:
            file_content2 = f.readlines()

        value_flow_lines1 = pair[0]["flow"]
        value_flow_lines2 = pair[1]["flow"]
        value_flow_raw1 = [
            file_content1[line - 1] for line in value_flow_lines1
        ]
        value_flow_raw2 = [
            file_content2[line - 1] for line in value_flow_lines2
        ]
        statements1 = strings_to_numpy(value_flow_raw1, self.__tokenizer,
                                       self.__config.encoder.name,
                                       self.__config.max_token_parts)
        value_flow1 = ValueFlow(statements=statements1,
                                n_statements=len(value_flow_raw1))
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
    [{"file": "dataset/project/commitid/files/*", "label":0, "flaws":[line,], "flows": [[line 1, line 2, ...], ...]}]
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
            self.__tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD),
                                            pad_token=PAD,
                                            length=config.max_token_parts)
            self.__tokenizer.enable_truncation(
                max_length=config.max_token_parts)
        elif config.encoder.name == "BERT":
            pass
        elif config.encoder.name == "HYBRID":
            pass
        else:
            raise ValueError(f"Can't find model: {self.encoder.name}")

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> MethodSample:
        method = self.__methods[index]
        n_flow = min(len(method["flows"]),
                     self.__config.hyper_parameters.max_n_flow)
        flow_indexes = numpy.arang(n_flow)
        numpy.random.shuffle(flow_indexes)

        value_flows = list()
        with open(method["file"], encoding="utf-8", errors="ignore") as f:
            file_content = f.readlines()
        for i in flow_indexes:
            value_flow_lines = method["flows"][i]
            value_flow_raw = [
                file_content[line - 1].strip() for line in value_flow_lines
            ]
            statements = strings_to_numpy(
                value_flow_raw, self.__tokenizer, self.__config.encoder.name,
                self.__config.encoder.max_token_parts)
            value_flows.append(
                ValueFlow(statements=statements,
                          n_statements=len(value_flow_raw),
                          statements_idx=value_flow_lines))

        return MethodSample(value_flows=value_flows,
                            label=method["label"],
                            flaws=method["flaws"])

    def get_n_samples(self):
        return self.__n_samples
