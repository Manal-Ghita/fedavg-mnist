import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 16x14x14
        x = self.pool(torch.relu(self.conv2(x)))  # 32x7x7
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x) 


if __name__ == "__main__":
    model = CNN()
    print("nb de poids :", sum(p.numel() for p in model.parameters()))
    for name, p in model.state_dict().items():
        print(name, tuple(p.shape))
