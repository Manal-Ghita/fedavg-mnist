import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def local_train(global_model, dataset, epochs=1, batch_size=32, lr=0.01):
    model = copy.deepcopy(global_model)  # le global reste intact
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    # len(dataset): poids dans la moyenne
    return model.state_dict(), len(dataset)
