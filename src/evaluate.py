import torch
from torch.utils.data import DataLoader


def evaluate(model, dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(dataset)
