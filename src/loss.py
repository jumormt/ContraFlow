from typing import Optional, Tuple, List
from src.utils import calc_sim_matrix, calc_strs_sim_matrix, calc_strs2_sim_matrix
import torch


def cross_entropy_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    pad_idx: Optional[int] = None,
    reduction: Optional[str] = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate cross entropy loss

    :param probabilities: [batch size; n classes] batch with logits
    :param target: [batch size; max classes] batch with padded class labels
    :param pad_idx: id of pad label
    :param reduction: how reduce a batch of losses, `None` mean no reduction
    :param eps: small value to avoid `log(0)`
    :return: loss
    """
    gathered_logits = torch.gather(probabilities, 1, target)
    if pad_idx is not None:
        pad_mask = target == pad_idx
        gathered_logits[pad_mask] = 1
    batch_loss = -(gathered_logits + eps).log().sum(-1)
    if reduction is None:
        return batch_loss
    elif reduction == "mean":
        return batch_loss.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def NCE_pair_loss(embeddings1: torch.Tensor,
                  embeddings2: torch.Tensor) -> torch.Tensor:
    """
    
    Args:
        embeddings1: [N; embed dim]
        embeddings2: [N; embed dim]

    Returns: loss

    """
    n_sample = embeddings1.size(0)
    # [N; N]
    sim_matrix = calc_sim_matrix(embeddings1, embeddings2)

    loss = 0
    for i in range(n_sample):
        positive_exp_sim_sum = torch.exp(sim_matrix[i][i])
        all_exp_sim_sum = torch.exp(sim_matrix[i]).sum()
        loss_i = positive_exp_sim_sum / all_exp_sim_sum
        loss = loss + loss_i
    sim_matrix_reverse = sim_matrix.transpose(0, 1)
    for i in range(n_sample):
        positive_exp_sim_sum = torch.exp(sim_matrix_reverse[i][i])
        all_exp_sim_sum = torch.exp(sim_matrix_reverse[i]).sum()
        loss_i = positive_exp_sim_sum / all_exp_sim_sum
        loss = loss + loss_i

    return loss / (2 * n_sample)


def NCE_loss(embeddings: torch.Tensor,
             sequences: List[Tuple[str, str]],
             features: torch.Tensor = None,
             thres: float = 0.5) -> torch.Tensor:
    """
    sum_i(sim(flow_i, [flow_i+])/sim(flow_i, [flow_i+, flow_i-])

    Args:
        embeddings (Tensor): [N; embed dim]
        sequences (List[Tuple[str, str]]): [(apis, types)]
        features (Tensor): [N; feature dim]
        thres (float): similarity threshold

    Returns: loss

    """
    n_sample = embeddings.size(0)
    # [N; N]
    sim_matrix = calc_sim_matrix(embeddings, embeddings)
    apis, types = zip(*sequences)
    # apis_sim = calc_strs_sim_matrix(apis, embeddings.get_device())
    # types_sim = calc_strs_sim_matrix(types, embeddings.get_device())

    sim = calc_strs2_sim_matrix(apis, types, embeddings.get_device())
    # sim_slice = (calc_sim_matrix(features, features) -
    #              torch.eye(n_sample, device=embeddings.get_device())) > thres
    # sim_slice = ((apis_sim + types_sim) / 2) > thres
    sim_slice = sim > thres
    loss = 0
    for i in range(n_sample):
        positive_exp_sim_sum = torch.exp(sim_matrix[i][sim_slice[i]]).sum()
        all_exp_sim_sum = torch.exp(sim_matrix[i]).sum()
        loss_i = positive_exp_sim_sum / all_exp_sim_sum
        loss = loss + loss_i
    return loss / sim_slice.sum()
