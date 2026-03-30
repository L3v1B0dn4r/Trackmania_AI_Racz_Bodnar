import torch
import numpy as np
import mss
import cv2
import controls
from model import DrivingModel
import time

time.sleep(0.03)

model = DrivingModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

# FIX 1: Matched the capture area back to training data ("top": 200)
monitor = {"top": 200, "left": 100, "width": 800, "height": 600}

print("AI Driver is starting...")

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ==========================================
        # FIX 2: Exact preprocessing from training
        # ==========================================
        frame = cv2.resize(frame, (160, 120))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
        frame = np.expand_dims(frame, axis=2)  # Add channel dimension: (120, 160) -> (120, 160, 1)
        # ==========================================

        input_data = frame / 255.0

        # permute(2,0,1) makes it (1, 120, 160)
        # unsqueeze(0) makes it (1, 1, 120, 160) to simulate a batch size of 1 for the model
        input_data = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        output = model(input_data)[0].detach().numpy()

        controls.release_all()

        # Assuming your action list was [A, D, W, S]
        if output[0] > 0.5:
            controls.left()  # 'left'
        if output[1] > 0.5:
            controls.right()  # 'right'
        if output[2] > 0.5:
            controls.accelerate()  # 'up'
        if output[3] > 0.5:
            controls.brake()  # 'down'