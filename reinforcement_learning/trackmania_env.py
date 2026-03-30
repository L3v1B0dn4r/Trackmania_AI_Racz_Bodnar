import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mss
import cv2
import time
import threading
import controls

# ==========================================
# FIX: Bypass TMInterface's strict signal handling
# so it doesn't crash in the background thread
# ==========================================
import signal

signal.signal = lambda *args, **kwargs: None
# ==========================================

# TMInterface Imports
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client


# ==========================================
# TMInterface Background Client
# ==========================================
class TMRewardClient(Client):
    def __init__(self):
        super().__init__()
        self.speed = 0
        self.is_finished = False
        self.iface = None

    def on_registered(self, iface: TMInterface):
        print("Successfully connected to TMInterface!")
        self.iface = iface

    def on_run_step(self, iface: TMInterface, time: int):
        # Continuously update the speed variable from the game's memory
        state = iface.get_simulation_state()
        self.speed = state.display_speed

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        # Check if the car hit the final checkpoint
        if current == target:
            self.is_finished = True


# ==========================================
# Gymnasium Environment
# ==========================================
class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super(TrackmaniaEnv, self).__init__()

        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8)

        self.sct = mss.mss()
        self.monitor = {"top": 200, "left": 100, "width": 800, "height": 600}

        self.step_counter = 0
        self.max_steps = 1000  # Give it enough time to drive

        # NEW: Track the previous speed to detect crashes
        self.prev_speed = 0

        # Start the TMInterface client in a background thread
        self.tm_client = TMRewardClient()
        self.tm_thread = threading.Thread(target=run_client, args=(self.tm_client,))
        self.tm_thread.daemon = True
        self.tm_thread.start()

        print("Waiting 3 seconds for TMInterface connection...")
        time.sleep(3)

    def get_observation(self):
        img = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (160, 120))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=2)
        return frame

    def step(self, action):
        self.step_counter += 1

        # 1. Apply Actions
        controls.release_all()
        if action[0] == 1: controls.left()
        if action[1] == 1: controls.right()
        if action[2] == 1: controls.accelerate()
        if action[3] == 1: controls.brake()

        time.sleep(0.1)  # Let physics play out

        # 2. Get new visual observation
        obs = self.get_observation()

        # 3. TMInterface REWARD FUNCTION
        current_speed = self.tm_client.speed

        # ==========================================
        # THE NEW REWARD & PENALTY SYSTEM
        # ==========================================
        reward = 0.0

        # 1. Base Reward: Going fast is good.
        reward += current_speed / 100.0

        # 2. Crash Penalty: If we were going fast and suddenly stopped, we hit a wall.
        if self.prev_speed > 50 and current_speed < 10:
            reward -= 50.0  # Big penalty for crashing

        # 3. Stuck Penalty: Standing still or reversing is bad.
        if current_speed < 5:
            reward -= 1.0  # Constant drain of points if it just sits there

        # 4. Wobble Penalty (Optional): Penalize pressing left and right at the same time
        if action[0] == 1 and action[1] == 1:
            reward -= 0.5

        # Update prev_speed for the next frame
        self.prev_speed = current_speed
        # ==========================================

        # 4. Check if Done
        terminated = False
        truncated = False

        # End episode if finished or time ran out
        if self.tm_client.is_finished:
            reward += 200.0  # Massive reward for completing the track
            terminated = True
        elif self.step_counter >= self.max_steps:
            truncated = True

        info = {"speed": current_speed}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0

        # Reset speed tracker so we don't accidentally penalize it for a "crash" at respawn
        self.prev_speed = 0
        self.tm_client.is_finished = False

        controls.release_all()

        # Instantly restart the map using TMInterface instead of pydirectinput
        if self.tm_client.iface:
            self.tm_client.iface.execute_command("delete")

        time.sleep(0.5)  # Short wait for respawn

        obs = self.get_observation()
        return obs, {}