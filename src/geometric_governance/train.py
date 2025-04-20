from typing import Literal
import torch
import torch.optim as o
from pydantic import BaseModel
from geometric_governance.model import ElectionResult


class TrainingSchema(BaseModel):
    num_epochs: int
    iterations_per_epoch: int
    learning_rate: float
    learning_rate_warmup_epochs: int
    learning_rate_warm_restart: bool
    clip_grad_norm: float


def compute_rule_loss(
    election: ElectionResult, winners: torch.Tensor, batch_size: int | None = None
):
    assert election.log_probs.shape == winners.shape
    batch_size = 1 if batch_size is None else batch_size
    return -(election.log_probs * winners).sum() / batch_size


def compute_welfare_loss(
    election: ElectionResult, welfare: torch.Tensor, batch_size: int | None = None
):
    assert election.log_probs.shape == welfare.shape
    batch_size = 1 if batch_size is None else batch_size
    return -(torch.exp(election.log_probs) * welfare).sum() / batch_size


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


def make_optim_and_scheduler(
    model: torch.nn.Module,
    lr: float,
    total_epochs: int | None = None,
    warmup_epochs: int = 5,
    warmup_start: float = 0.1,
    warmup_end: float = 1,
    T_0: int = 5,
    T_mult: int = 2,
    warm_restart: bool = True,
):
    optim = o.Adam(model.parameters(), lr=lr)

    warmup_scheduler = o.lr_scheduler.LinearLR(
        optim,
        start_factor=warmup_start,
        end_factor=warmup_end,
        total_iters=warmup_epochs,
    )
    if warm_restart:
        main_scheduler: o.lr_scheduler.LRScheduler = (
            o.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=T_mult)
        )
    else:
        assert total_epochs is not None
        main_scheduler = o.lr_scheduler.CosineAnnealingLR(
            optim, total_epochs - warmup_epochs
        )
    scheduler = o.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    return optim, scheduler
