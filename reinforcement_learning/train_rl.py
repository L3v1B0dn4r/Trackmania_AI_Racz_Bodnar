import time
import keyboard  # NEW: Added for the global kill switch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# THIS IS THE CRITICAL LINE THAT IS MISSING
from trackmania_env import TrackmaniaEnv

# --- NEW: Global Kill Switch Callback ---
class GlobalKillSwitchCallback(BaseCallback):
    def __init__(self, stop_key="q", verbose=0):
        super().__init__(verbose)
        self.stop_key = stop_key

    def _on_step(self) -> bool:
        # Check if the global hotkey is pressed
        if keyboard.is_pressed(self.stop_key):
            print(f"\n[!] Kill switch activated (pressed '{self.stop_key}'). Stopping training gracefully...")
            return False  # Returning False tells SB3 to stop the learn() loop
        return True
# ----------------------------------------

print("Starting RL Training...")
print("Focus the Trackmania window in 5 seconds!")
time.sleep(5)

# 1. Create the environment
env = TrackmaniaEnv()

# 2. Custom Hyperparameters for Vision-based Racing
custom_ppo_args = {
    "learning_rate": 1e-4,  # Slower, more stable learning for CNNs processing images
    "n_steps": 1024,  # Learn after 1024 steps of driving (~100 seconds)
    "batch_size": 64,  # Study 64 frames at a time
    "gamma": 0.995,  # Look further ahead into the future (important for braking)
    "ent_coef": 0.01,  # Force the AI to experiment with steering and braking

# learning_rate (Default: 0.0003): Controls how drastically the AI updates its brain.
# We lower it to 1e-4 because the AI is processing complex images (CNNs),
# making the learning process more stable and less prone to "forgetting" how to drive.

# n_steps (Default: 2048): The number of steps the AI takes before it pauses to learn.
# 1024 steps at ~10 actions per second equals roughly 100 seconds of driving,
# which is a good chunk of track to learn from before updating weights.

# batch_size (Default: 64): How many experiences it studies at once from memory.
# 64 is a good balance for processing images. If your computer's RAM or VRAM
# gets overloaded, you can lower this to 32.

# gamma (Default: 0.99): The "discount factor" for future rewards.
# A higher gamma like 0.995 forces the AI to look further into the future,
# which is crucial for racing so it learns to brake *now* to avoid a wall *later*.

# ent_coef (Default: 0.0): The "entropy coefficient".
# Setting this to 0.01 forces the AI to experiment with random actions (like steering).
# Without it, the AI might just hold 'Accelerate' forever if it finds a tiny reward early on.
}

print("Initializing PPO Model...")

# 3. Initialize the PPO algorithm with a CNN policy
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_trackmania_tensorboard/",
    **custom_ppo_args
)

# 4. Set up Auto-Saving (Checkpoints)
# This saves a backup of your AI's brain every 10,000 steps.
# The files will be saved in a new folder called "checkpoints".
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="tm_ppo_model"
)

# Initialize your new kill switch
kill_switch_callback = GlobalKillSwitchCallback(stop_key="q")

# 5. Train the agent
try:
    print("=====================================================")
    print("Training started! Press 'q' ANYWHERE to stop and save early.")
    print("=====================================================")

    # Start the learning process and pass BOTH callbacks as a list
    model.learn(
        total_timesteps=100000,
        callback=[checkpoint_callback, kill_switch_callback]
    )

except KeyboardInterrupt:
    print("\nTraining interrupted via Terminal (Ctrl+C).")

# 6. Save the final model and clean up
print("\nSaving current progress...")
model.save("ppo_trackmania_model_final")
print("Final model saved as ppo_trackmania_model_final.zip!")

env.close()