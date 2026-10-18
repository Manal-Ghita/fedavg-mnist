import copy

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class TriggerSet(Dataset):
    # images de bruit + labels au hasard = la clé secrète
    def __init__(self, n=100, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 1, 28, 28, generator=g)
        self.y = torch.randint(0, 10, (n,), generator=g).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def train(model, dataset, epochs=3, batch_size=64, lr=0.01):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return model


def train_with_watermark(model, train_set, trigger, epochs=3, repeat=20):
    # trigger répété sinon il est noyé dans les 60k images
    mixed = ConcatDataset([train_set] + [trigger] * repeat)
    return train(model, mixed, epochs)


def quantize_int8(model):
    # quantif int8 symétrique par tenseur, comme sur MCU
    q = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in q.named_parameters():
            if "weight" in name:
                scale = p.abs().max() / 127
                p.copy_(torch.round(p / scale).clamp(-127, 127) * scale)
    return q


def prune_model(model, amount=0.5):
    # on met à zéro les plus petits poids
    p = copy.deepcopy(model)
    for m in p.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, "weight", amount=amount)
            prune.remove(m, "weight")
    return p


def finetune(model, dataset, epochs=1):
    # attaque : quelqu'un réentraîne le modèle volé sur ses données
    return train(copy.deepcopy(model), dataset, epochs, lr=0.001)