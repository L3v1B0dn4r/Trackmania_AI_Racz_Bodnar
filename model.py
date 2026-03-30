import torch
import torch.nn as nn

class DrivingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(48 * 17 * 12, 100),
            nn.ReLU(),
            nn.Linear(100, 4),
            nn.Sigmoid()  # IMPORTANT
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)