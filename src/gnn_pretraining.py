from argparse import ArgumentParser
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from transformers import RobertaTokenizer
from src.datas.datamodules import ASTDataModule
from src.models.gnn_pretraining import GNNPretraining
from src.train import train
from src.utils import filter_warnings


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            type=str)
    return arg_parser


def pretrain(config_path: str):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))
    print_config(config, ["ast", "hyper_parameters"])
    seed_everything(config.seed, workers=True)

    # Load tokenizer

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id
    # Init datamodule
    data_module = ASTDataModule(config, tokenizer)

    # Init model
    model = GNNPretraining(config, vocab_size, pad_idx)

    train(model, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    pretrain(__args.config)
