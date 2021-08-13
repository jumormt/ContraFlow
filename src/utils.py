from warnings import filterwarnings
import subprocess
from tokenizers import Tokenizer
from typing import List, Union, Dict, Tuple
import numpy
import torch
from transformers import RobertaTokenizer
import os
from src.sequence_analyzer.sequence_analyzer import SequencesAnalyzer

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.trainer.data_loading",
                   lineno=102)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.utilities.data",
                   lineno=41)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=216)  # save
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=234)  # load
    filterwarnings("ignore",
                   category=DeprecationWarning,
                   module="pytorch_lightning.metrics",
                   lineno=43)


def count_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path],
                                    capture_output=True,
                                    encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(
            f"Counting lines in {file_path} failed with error\n{command_result.stderr}"
        )
    return int(command_result.stdout.split()[0])


def strings_to_numpy(values: List[str], tokenizer: Union[Tokenizer,
                                                         RobertaTokenizer],
                     encoder_name: str, max_len: int) -> numpy.ndarray:
    """
    transform a list of long strings to numpy array using tokenizer

    Args:
        values (List[str]): a list of long strings, e.g., ["int i = 1;", "return i;"]
        tokenizer:
        encoder_name (str): the name of encoder
        max_len (int): the max len to encode each long string

    Returns:
        : [max_len; len(values)], dtype=numpy.compat.long
    """
    res = numpy.full((max_len, len(values)),
                     tokenizer.pad_token_id,
                     dtype=numpy.compat.long)

    for i, value in enumerate(values):
        if encoder_name in ["BERT", "HYBRID"]:
            tokens = [tokenizer.cls_token
                      ] + tokenizer.tokenize(value) + [tokenizer.sep_token]
        elif encoder_name in ["LSTM", "GNN"]:
            tokens = tokenizer.tokenize(value)
        else:
            raise ValueError(f"Cant find encoder name: {encoder_name}!")
        ids = tokenizer.convert_tokens_to_ids(tokens)
        less_len = min(len(ids), max_len)
        res[:less_len, i] = ids[:less_len]

    # if encoder_name == "LSTM":
    #     res = numpy.full((max_len, len(values)),
    #                      tokenizer.token_to_id(PAD),
    #                      dtype=numpy.compat.long)

    #     for i, value in enumerate(values):
    #         res[:, i] = tokenizer.encode(value).ids
    # elif encoder_name in ["BERT", "GNN", "HYBRID"]:
    #     res = numpy.full((max_len, len(values)),
    #                      tokenizer.pad_token_id,
    #                      dtype=numpy.compat.long)
    #     for i, value in enumerate(values):
    #         tokens = [tokenizer.cls_token
    #                   ] + tokenizer.tokenize(value) + [tokenizer.sep_token]
    #         ids = tokenizer.convert_tokens_to_ids(tokens)
    #         less_len = min(len(ids), max_len)
    #         res[:less_len, i] = ids[:less_len]
    # else:
    #     raise ValueError(f"Cant find encoder name: {encoder_name}!")
    return res


def calc_sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """
    caculate the cosine similarity between each vector in a and b

    Args:
        a (Tensor): [N; dim]
        b (Tensor): [N; dim]
        eps (float): avoid numerical error

    Returns: [N; N]
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def segment_sizes_to_slices(sizes: torch.Tensor) -> List:
    """convert length per sample list to slice

    Args:
        sizes (Tensor): [n_sample]

    Returns:
        : List[slice(start, end)]

    Examples::
        [1,2,3] -> [(0, 1), (1, 3), (3, 6)]
    """
    cum_sums = numpy.cumsum(sizes.cpu())
    start_of_segments = numpy.append([0], cum_sums[:-1])
    return [
        slice(start, end) for start, end in zip(start_of_segments, cum_sums)
    ]


def read_csv(csv_file_path: str) -> List[Dict]:
    """
        read csv to memory
    Args:
        csv_file_path (str): path to csv

    Returns:
        : List[row]

    """
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


def get_ast_path_from_file(file_path: str):
    """

    Args:
        file_path (str): e.g., dataset/project/commitid/files/test.cpp

    get node.csv and edge.csv from source file path
    """
    tmp, file_name = os.path.split(file_path)
    tmp, _ = os.path.split(tmp)
    root, _ = os.path.split(tmp)
    node_path = os.path.join(root, f"graphs/{file_name}/nodes.csv")
    edge_path = os.path.join(root, f"graphs/{file_name}/edges.csv")
    return node_path, edge_path


def cut_lower_embeddings(
        lower_embeddings: torch.Tensor,
        lower_per_upper: torch.Tensor,
        mask_value: float = -1e9) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut lower embeddings into upper embeddings

    Args:
        lower_embeddings (Tensor): [total n_lower; units]
        statements_per_label (Tensor): [n_upper]
        mask_value (float): -inf

    Returns: [n_upper; max n_lower; units], [n_upper; max n_lower]
    """
    batch_size = len(lower_per_upper)
    max_context_len = max(lower_per_upper)

    upper_embeddings = lower_embeddings.new_zeros(
        (batch_size, max_context_len, lower_embeddings.shape[-1]))
    upper_attn_mask = lower_embeddings.new_zeros((batch_size, max_context_len))

    statments_slices = segment_sizes_to_slices(lower_per_upper)
    for i, (cur_slice,
            cur_size) in enumerate(zip(statments_slices, lower_per_upper)):
        upper_embeddings[i, :cur_size] = lower_embeddings[cur_slice]
        upper_attn_mask[i, cur_size:] = mask_value

    return upper_embeddings, upper_attn_mask


def calc_strs_sim_matrix(strs: List[str],
                         device: torch.device) -> torch.Tensor:
    """
    caculate the sequence similarity between each vector in a and b

    Args:
        strs (List[str]): [N] a list of strings
        device:

    Returns:
        [N; N]
    """
    N = len(strs)
    res = torch.zeros(N, N, dtype=torch.float, device=device)
    for i in range(N):
        for j in range(i + 1, N):

            res[i][j] = res[j][i] = SequencesAnalyzer(strs[i],
                                                      strs[j]).similarity()
    return res


def calc_strs2_sim_matrix(apis: List[str], types: List[str],
                          device: torch.device) -> torch.Tensor:
    """
    caculate the sequence similarity between each vector in a and b

    Args:
        apis (List[str]): [N] a list of apis
        types (List[str]): [N] a list of types
        device:

    Returns:
        [N; N]
    """
    N = len(apis)
    res = torch.zeros(N, N, dtype=torch.float, device=device)
    for i in range(N):
        for j in range(i + 1, N):
            if (len(apis[i]) == 0 or len(apis[j]) == 0):
                if (len(types[i]) != 0 and len(types[j]) != 0):
                    res[i][j] = res[j][i] = SequencesAnalyzer(
                        types[i], types[j]).similarity()
                else:
                    res[i][j] = res[j][i] = 0
            else:
                if (len(types[i]) != 0 and len(types[j]) != 0):
                    res[i][j] = res[j][i] = (
                        SequencesAnalyzer(apis[i], apis[j]).similarity() +
                        SequencesAnalyzer(types[i], types[j]).similarity()) / 2
                else:
                    res[i][j] = res[j][i] = SequencesAnalyzer(
                        apis[i], apis[j]).similarity()
    return res
