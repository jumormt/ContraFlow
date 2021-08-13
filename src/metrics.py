from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import numpy


@dataclass
class Statistic:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    BTP_P: List = field(default_factory=list)
    BTP_R: List = field(default_factory=list)
    BTP_F1: List = field(default_factory=list)
    # overlap, union, predicted, label
    n_lines: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def update(self, other_statistic: "Statistic"):
        self.true_positive += other_statistic.true_positive
        self.false_positive += other_statistic.false_positive
        self.false_negative += other_statistic.false_negative
        self.true_negative += other_statistic.true_negative
        self.BTP_R += other_statistic.BTP_R
        self.BTP_P += other_statistic.BTP_P
        self.BTP_F1 += other_statistic.BTP_F1
        self.n_lines += other_statistic.n_lines

    def calculate_metrics(self, group: Optional[str] = None) -> Dict[str, int]:
        precision, recall, f1, fpr, acc = 0, 0, 0, 0, 0
        acc = (self.true_negative + self.true_positive) / (
            self.true_positive + self.true_negative + self.false_positive +
            self.false_negative)
        if self.true_positive + self.false_positive > 0:
            precision = self.true_positive / (self.true_positive +
                                              self.false_positive)
        if self.false_positive + self.true_negative > 0:
            fpr = self.false_positive / (self.false_positive +
                                         self.true_negative)
        if self.true_positive + self.false_negative > 0:
            recall = self.true_positive / (self.true_positive +
                                           self.false_negative)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        metrics_dict = {
            "fpr": fpr.item() if type(fpr) == torch.Tensor else fpr,
            "precision":
            precision.item() if type(precision) == torch.Tensor else precision,
            "recall":
            recall.item() if type(recall) == torch.Tensor else recall,
            "accuracy": acc.item() if type(acc) == torch.Tensor else acc,
            "f1": f1.item() if type(f1) == torch.Tensor else f1
        }
        if group is not None:
            for key in list(metrics_dict.keys()):
                metrics_dict[f"{group}_{key}"] = metrics_dict.pop(key)
        return metrics_dict

    @staticmethod
    def calculate_statistic(labels: torch.Tensor, preds: torch.Tensor,
                            nb_classes: int) -> "Statistic":
        """Calculate statistic for ground truth and predicted batches of labels.
        :param labels: ground truth labels
        :param preds: predicted labels
        :param skip: list of subtokens ids that should be ignored
        :return: dataclass with calculated statistic
        """
        statistic = Statistic()
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        statistic.true_negative, statistic.false_positive, statistic.false_negative, statistic.true_positive = confusion_matrix[
            0,
            0], confusion_matrix[0,
                                 1], confusion_matrix[1,
                                                      0], confusion_matrix[1,
                                                                           1]
        return statistic

    def mean_BTP(self, group: Optional[str] = None) -> Dict[str, float]:
        if (len(self.BTP_P) == 0):
            metrics_dict = {"BTP_P": 0.0, "BTP_R": 0.0, "BTP_F1": 0.0}
        else:
            metrics_dict = {
                "BTP_P": numpy.mean(self.BTP_P),
                "BTP_R": numpy.mean(self.BTP_R),
                "BTP_F1": numpy.mean(self.BTP_F1)
            }
        if group is not None:
            for key in list(metrics_dict.keys()):
                metrics_dict[f"{group}_{key}"] = metrics_dict.pop(key)
        return metrics_dict

    def calc_btp_metrics(self,
                         true_positives_slice: torch.Tensor,
                         statements_idxes: List,
                         flaws: List,
                         method_weights: torch.Tensor,
                         flow_weights: torch.Tensor,
                         group: Optional[str] = None,
                         k: int = 3) -> Dict[str, float]:
        """
        calculate BTP metrics

        Args:
            true_positives_slice (Tensor): [n_method]
            statements_idxes (List): [n_method (list)] -> [n_value_flow (list)] -> [n_statements]
            flaws (List): [n_method (list)] -> [n_statements]
            method_weights (Tensor): [n_method, max n_flow]
            flow_weights (Tensor): [n_method, max n_flow; max flow n_statements]
            group (str): train/val/test
            k (int):
        
        Returns:
            {
                "BTP_P": numpy.mean(self.BTP_P),
                "BTP_R": numpy.mean(self.BTP_R),
                "BTP_F1": numpy.mean(self.BTP_F1)
            }
        """
        # no true positive samples
        if (sum(true_positives_slice) == 0):
            return self.mean_BTP(group)
        statements_idxes, flaws = numpy.array(
            statements_idxes)[true_positives_slice], numpy.array(flaws)
        method_weights, flow_weights = method_weights[
            true_positives_slice], flow_weights[true_positives_slice]
        # [n_method, k]
        _, topk_values = torch.topk(method_weights, k, dim=-1)

        # for each method
        for i in range(flow_weights.size(0)):
            predicted_idxes = set()
            statements_idxes[i] = numpy.array(statements_idxes[i])
            # [k'] k' value flows
            topk_values_idx = topk_values[i][
                topk_values[i] < len(statements_idxes[i])]
            value_flows_st_idxes = statements_idxes[i][topk_values_idx]

            # [k'; max flow n_statements]
            flow_statements = flow_weights[i][topk_values_idx]
            # [k'; k]
            _, topk_statements = torch.topk(flow_statements, k, dim=-1)
            # for each value flow
            for j in range(flow_statements.size(0)):
                value_flows_st_idxes[j] = numpy.array[value_flows_st_idxes[j]]
                # [k''] k'' statements
                topk_statements_idx = topk_statements[j][
                    topk_statements[j] < len(value_flows_st_idxes[j])]
                predicted_idxes = predicted_idxes.union(
                    set(value_flows_st_idxes[j][topk_statements_idx]))
            true_idxes = set(flaws[i])
            n_overlap = len(true_idxes.intersection(predicted_idxes))
            n_union = len(true_idxes.union(predicted_idxes))
            self.n_lines.append(
                (n_overlap, n_union, len(predicted_idxes), len(true_idxes)))
            BTP_P = n_overlap / len(predicted_idxes)
            BTP_R = n_overlap / len(true_idxes)
            self.BTP_P.append(BTP_P)
            self.BTP_R.append(BTP_R)
            if (BTP_P + BTP_R != 0):
                self.BTP_F1.append(BTP_P * BTP_R / (BTP_P + BTP_R))
            else:
                self.BTP_F1.append(0.0)
        return self.mean_BTP(group)

    @staticmethod
    def union_statistics(stats: List["Statistic"]) -> "Statistic":
        union_statistic = Statistic()
        for stat in stats:
            union_statistic.update(stat)
        return union_statistic
