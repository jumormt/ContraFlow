from warnings import filterwarnings
import subprocess
from tokenizers import Tokenizer
from typing import List
import numpy
import torch

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


def count_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path],
                                    capture_output=True,
                                    encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(
            f"Counting lines in {file_path} failed with error\n{command_result.stderr}"
        )
    return int(command_result.stdout.split()[0])


def strings_to_numpy(values: List[str], tokenizer: Tokenizer,
                     max_len: int) -> numpy.ndarray:
    res = numpy.full((max_len, len(values)),
                     tokenizer.token_to_id(PAD),
                     dtype=numpy.long)

    for i, value in enumerate(values):
        res[:, i] = tokenizer.encode(value).ids
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
