from typing import Literal
import torch
from geometric_governance.model import ElectionResult


def compute_rule_loss(
    election: ElectionResult, winners: torch.Tensor, batch_size: int | None = None
):
    assert election.log_probs.shape == winners.shape
    batch_size = 1 if batch_size is None else batch_size
    return -(election.log_probs * winners).sum() / batch_size


def compute_monotonicity_loss(
    election: ElectionResult,
    data,
    batch_size: int,
    representation: Literal["graph", "set"] = "graph",
):
    if representation == "set":
        raise NotImplementedError(
            "Monotonicity loss not implemented for set representation."
        )
    candidates = data.candidate_idxs.nonzero()
    perm = torch.randperm(candidates.size(0))[:batch_size]
    loss: torch.Tensor = 0.0  # type: ignore
    for i in perm:
        candidate_idx = candidates[i]
        edge_idxs = data.edge_index[1] == candidate_idx
        grad = torch.autograd.grad(
            outputs=torch.exp(election.log_probs[i]),
            inputs=data.edge_attr,
            create_graph=True,
        )[0]
        loss += torch.where(
            grad[edge_idxs] < 0,
            -grad[edge_idxs],
            torch.zeros_like(grad[edge_idxs]),
        ).mean()
    loss /= batch_size
    return loss
