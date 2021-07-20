from pytorch_lightning import LightningDataModule
from tokenizers import Tokenizer
from omegaconf import DictConfig
from os import cpu_count
from os.path import join

import torch

from src.datas.datastructures import ValueFlowPairBatch, ValueFlowPair, MethodSampleBatch, MethodSample, ValueFlow, ValueFlowBatch
from src.datas.datasets import ValueFlowPairDataset, MethodSampleDataset, ValueFlowDataset
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset


class ValueFlowDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, tokenizer: Tokenizer
                 ):
        super().__init__()
        self.__tokenizer = tokenizer
        self.__config = config
        self.__data_folder = config.data_folder
        self.__n_workers = cpu_count(
        ) if self.__config.num_workers == -1 else self.__config.num_workers

    @staticmethod
    def collate_wrapper(batch: List[ValueFlow]) -> ValueFlowBatch:
        return ValueFlowBatch(batch)

    def __create_dataset(self, data_path: str) -> Dataset:
        return ValueFlowDataset(data_path, self.__tokenizer, self.__config)

    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__data_folder, "train.json")
        train_dataset = self.__create_dataset(train_dataset_path)
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__data_folder, "val.json")
        val_dataset = self.__create_dataset(val_dataset_path)
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset_path = join(self.__data_folder, "test.json")
        test_dataset = self.__create_dataset(test_dataset_path)
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(
            self,
            batch: ValueFlowBatch,
            device: Optional[torch.device] = None) -> ValueFlowBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch


class ValueFlowPairDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, tokenizer: Tokenizer
                 ):
        super().__init__()
        self.__tokenizer = tokenizer
        self.__config = config
        self.__data_folder = config.data_folder
        self.__n_workers = cpu_count(
        ) if self.__config.num_workers == -1 else self.__config.num_workers

    @staticmethod
    def collate_wrapper(batch: List[ValueFlowPair]) -> ValueFlowPairBatch:
        return ValueFlowPairBatch(batch)

    def __create_dataset(self, data_path: str) -> Dataset:
        return ValueFlowPairDataset(data_path, self.__tokenizer, self.__config)

    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__data_folder, "train.json")
        train_dataset = self.__create_dataset(train_dataset_path)
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__data_folder, "val.json")
        val_dataset = self.__create_dataset(val_dataset_path)
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset_path = join(self.__data_folder, "test.json")
        test_dataset = self.__create_dataset(test_dataset_path)
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(
            self,
            batch: ValueFlowPairBatch,
            device: Optional[torch.device] = None) -> ValueFlowPairBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch


class MethodSampleDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, tokenizer: Tokenizer
                 ):
        super().__init__()
        self.__tokenizer = tokenizer
        self.__config = config
        self.__data_folder = config.data_folder
        self.__n_workers = cpu_count(
        ) if self.__config.num_workers == -1 else self.__config.num_workers

    @staticmethod
    def collate_wrapper(batch: List[MethodSample]) -> MethodSampleBatch:
        return MethodSampleBatch(batch)

    def __create_dataset(self, data_path: str) -> Dataset:
        return MethodSampleDataset(data_path, self.__tokenizer, self.__config)

    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__data_folder, "train.json")
        train_dataset = self.__create_dataset(train_dataset_path)
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__data_folder, "val.json")
        val_dataset = self.__create_dataset(val_dataset_path)
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset_path = join(self.__data_folder, "test.json")
        test_dataset = self.__create_dataset(test_dataset_path)
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(
            self,
            batch: MethodSampleBatch,
            device: Optional[torch.device] = None) -> MethodSampleBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch
