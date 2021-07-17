from dataclasses import dataclass
from typing import List

import numpy
import torch


@dataclass
class ValueFlow:
    statements: numpy.ndarray  # [seq len; n_statements]
    n_statements: int
    raw_statements: List[str]
    feature: numpy.ndarray = None  # [feature dim,]


class ValueFlowBatch:
    def __init__(self, value_flows: List[ValueFlow]):
        # [batch size]
        self.statements_per_label = torch.tensor(
            [value_flow.n_statements for value_flow in value_flows],
            dtype=torch.long)
        self.raw_statements = [
            value_flow.raw_statements for value_flow in value_flows
        ]

        # [seq len; total n_statements]
        self.statements = torch.from_numpy(
            numpy.hstack([value_flow.statements
                          for value_flow in value_flows]))

        # [batch size, feature dim]
        self.features = numpy.full(
            (len(value_flows), value_flows[0].feature.shape[0]),
            0,
            dtype=numpy.long)

        for i, value_flow in enumerate(value_flows):
            self.features[i] = value_flow.feature
        self.features = torch.from_numpy(self.features)

    def __len__(self):
        return len(self.statements_per_label)

    def pin_memory(self) -> "ValueFlowBatch":
        self.statements_per_label = self.statements_per_label.pin_memory()
        self.statements = self.statements.pin_memory()
        self.features = self.features.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.statements_per_label = self.statements_per_label.to(device)
        self.statements = self.statements.to(device)
        self.features = self.features.to(device)


@dataclass
class ValueFlowPair:
    value_flow_1: ValueFlow
    value_flow_2: ValueFlow


class ValueFlowPairBatch:
    def __init__(self, value_flow_pairs: List[ValueFlowPair]):
        # [batch size]
        self.statements_per_label1 = torch.tensor(
            [_pair.value_flow_1.n_statements for _pair in value_flow_pairs],
            dtype=torch.long)
        self.statements_per_label2 = torch.tensor(
            [_pair.value_flow_2.n_statements for _pair in value_flow_pairs],
            dtype=torch.long)
        # self.raw_statements1 = [_pair.value_flow_1.raw_statements for _pair in value_flow_pairs]

        # [seq len; total n_statements]
        self.statements1 = torch.from_numpy(
            numpy.hstack(
                [_pair.value_flow_1.statements for _pair in value_flow_pairs]))
        self.statements2 = torch.from_numpy(
            numpy.hstack(
                [_pair.value_flow_2.statements for _pair in value_flow_pairs]))
        # self.raw_statements2 = [_pair.value_flow_2.raw_statements for _pair in value_flow_pairs]

    def __len__(self):
        return len(self.statements_per_label1)

    def pin_memory(self) -> "ValueFlowPairBatch":
        self.statements_per_label1 = self.statements_per_label1.pin_memory()
        self.statements_per_label2 = self.statements_per_label2.pin_memory()
        self.statements1 = self.statements1.pin_memory()
        self.statements2 = self.statements2.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.statements_per_label1 = self.statements_per_label1.to(device)
        self.statements_per_label2 = self.statements_per_label1.to(device)
        self.statements1 = self.statements1.to(device)
        self.statements2 = self.statements2.to(device)


@dataclass
class MethodSample:
    value_flows: List[ValueFlow]
    label: int


class MethodSampleBatch:
    def __init__(self, method_samples: List[MethodSample]):
        # [batch size]
        self.value_flow_per_label = torch.tensor(
            [len(method_sample) for method_sample in method_samples],
            dtype=torch.long)
        self.labels = torch.tensor(
            [method_sample.label for method_sample in method_samples],
            dtype=torch.long)

        self.statements_per_value_flow = list()
        self.value_flows = list()
        self.methods_raw: List[List[List[str]]] = list()
        for method_sample in method_samples:
            self.statements_per_value_flow.extend([
                value_flow.n_statements
                for value_flow in method_sample.value_flows
            ])
            self.value_flows.append(
                # [seq len; method n_statements]
                torch.from_numpy(
                    numpy.hstack([
                        value_flow.statements
                        for value_flow in method_sample.value_flows
                    ])))
            self.methods_raw.append([
                value_flow.raw_statements
                for value_flow in method_sample.value_flows
            ])
        # [seq len; total n_statements]
        self.statements = torch.cat(self.value_flows, dim=1)
        # [total n_flow]
        self.statements_per_value_flow = torch.tensor(
            self.statements_per_value_flow, dtype=torch.long)

    def __len__(self):
        return len(self.value_flow_per_label)

    def pin_memory(self) -> "MethodSampleBatch":
        self.value_flow_per_label = self.value_flow_per_label.pin_memory()
        self.statements_per_value_flow = self.statements_per_value_flow.pin_memory(
        )
        self.statements = self.statements.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.value_flow_per_label = self.value_flow_per_label.to(device)
        self.statements_per_value_flow = self.statements_per_value_flow.to(
            device)
        self.statements = self.statements.to(device)
        self.labels = self.labels.to(device)
