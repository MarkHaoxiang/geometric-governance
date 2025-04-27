import torch
from torch_geometric.loader import DataLoader
from geometric_governance.objective import VotingObjectiveRegistry


def get_mechanism_welfare(mechanism, dataloader: DataLoader):
    dataset = dataloader.dataset
    assert len(dataset) > 0
    assert 