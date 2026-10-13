import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def load_mnist(root="../data"):
    tf = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return train, test


def split_iid(dataset, n_clients=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g).tolist()

    size = len(dataset) // n_clients  # le reste est ignoré
    return [Subset(dataset, idx[i * size:(i + 1) * size]) for i in range(n_clients)]
