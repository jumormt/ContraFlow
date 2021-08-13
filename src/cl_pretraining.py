from argparse import ArgumentParser
from os.path import join, basename
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from tokenizers import Tokenizer
from transformers import RobertaTokenizer
from src.datas.datamodules import ValueFlowDataModule
from src.models.cl_pretraining import FlowCLPretraining
from src.train import train
from src.utils import filter_warnings, PAD


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
    print_config(config, ["encoder", "hyper_parameters"])
    seed_everything(config.seed, workers=True)

    # Load tokenizer
    if config.encoder.name == "LSTM":
        # dataset_name = basename(config.data_folder)
        # tokenizer_path = join(f"../data/tokenizer/{dataset_name}",
        #                       "tokenizer.json")
        # tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        vocab_size = tokenizer.vocab_size
        pad_idx = tokenizer.pad_token_id
    elif config.encoder.name in ["BERT", "GNN", "HYBRID"]:
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        vocab_size = tokenizer.vocab_size
        pad_idx = tokenizer.pad_token_id
    else:
        raise ValueError(f"Can't find encoder: {config.encoder.name}")

    # Init datamodule
    data_module = ValueFlowDataModule(config, tokenizer)

    # Init model
    model = FlowCLPretraining(config, vocab_size, pad_idx,
                              config.encoder.pretrained)

    train(model, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    pretrain(__args.config)
