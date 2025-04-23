import torch
from torch_geometric.loader import DataLoader


def get_mechanism_welfare(mechanism, dataloader: DataLoader):
    data = dataloader.dataset
