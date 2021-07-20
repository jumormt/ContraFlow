from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from typing import cast

from itertools import chain, repeat
from typing import Counter as TypeCounter
from collections import Counter

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from src.datas.preprocess.lexical_parser import tokenize_code_line
from src.utils import UNK, MASK, PAD, count_lines_in_file, filter_warnings

DROPOUT = None
VOCAB_SIZE = 100000
SPECIAL_TOKENS = [PAD, UNK, MASK]


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            type=str)
    return arg_parser


def batch_iterator(subtokens_counter: Counter[str]) -> chain[str]:
    return chain(*(repeat(st.encode("utf-8", "ignore").decode("utf-8"), cnt)
                   for st, cnt in subtokens_counter.items()))


def train_bpe(config_path: str):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))
    token_counter: TypeCounter[str] = Counter()

    # create token counter from tokens.txt
    with open(config.tokens_path, "r") as f:
        for line in tqdm(f, total=count_lines_in_file(config.tokens_path)):
            # tokens = line.split()
            tokens = tokenize_code_line(line, config.subtoken)
            token_counter.update(tokens)

    bpe_tokenizer = Tokenizer(
        BPE(dropout=DROPOUT, unk_token=UNK, fuse_unk=True))

    bpe_tokenizer.normalizer = Sequence([NFKC()])
    bpe_tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    length = sum([cnt for _, cnt in token_counter.items()])
    bpe_tokenizer.train_from_iterator(batch_iterator(token_counter),
                                      trainer=trainer,
                                      length=length)

    bpe_tokenizer.save(config.tokenizer_path)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()

    train_bpe(__args.config)
