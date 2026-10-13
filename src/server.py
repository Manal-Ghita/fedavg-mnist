import torch

from .model import CNN
from .client import local_train
from .evaluate import evaluate


def fedavg(updates):
    total = sum(n for _, n in updates)

    new_state = {}
    for key in updates[0][0]:
        new_state[key] = sum(state[key] * n for state, n in updates) / total
    return new_state


def run_fedavg(clients, test, n_rounds, local_epochs=1, frac=1.0, seed=0):
    torch.manual_seed(seed)
    global_model = CNN()
    g = torch.Generator().manual_seed(seed)  # tirage des clients
    k = max(1, round(frac * len(clients)))

    accs = []
    for _ in range(n_rounds):
        chosen = torch.randperm(len(clients), generator=g)[:k].tolist()
        updates = [local_train(global_model, clients[i], epochs=local_epochs) for i in chosen]
        global_model.load_state_dict(fedavg(updates))
        accs.append(evaluate(global_model, test))
    return accs
