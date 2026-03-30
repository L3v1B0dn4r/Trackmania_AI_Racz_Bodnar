import numpy as np
import torch
import torch.nn as nn
from model import DrivingModel

data = np.load("training_data.npy", allow_pickle=True)

X = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

X = X / 255.0

X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y = torch.tensor(y, dtype=torch.float32)

model = DrivingModel()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ep = 2      # Epoch number

for epoch in range(ep):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pth")