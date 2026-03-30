from stable_baselines3 import PPO
from trackmania_env import TrackmaniaEnv
import time
import controls

print("Loading trained AI...")
model = PPO.load("ppo_trackmania_model")

env = TrackmaniaEnv()
obs, _ = env.reset()

print("AI Driver is starting in 3 seconds. Focus the game!")
time.sleep(3)

try:
    while True:
        # The model predicts the best action based on the screenshot
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

except KeyboardInterrupt:
    print("Stopping AI.")
finally:
    controls.release_all()
    env.close()