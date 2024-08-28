import gymnasium as gym
from stable_baselines3 import PPO
import cv2
import os
import numpy as np
import flappy_bird_gymnasium 

env = gym.make("FlappyBird-v0", render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

num_episodes = 1000
for episode in range(1, num_episodes + 1):
    print(f"episode:{episode}")
    obs, _= env.reset()
    episode_path = f"data/episode{episode}"
    os.makedirs(episode_path, exist_ok=True)
    actions = []

    t = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs ,reward, done, _, _ = env.step(action )

        img = env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 转换为BGR格式以适用于OpenCV
        cv2.imwrite(os.path.join(episode_path, f"{t + 1}.png"), img)

        actions.append(action)
        t += 1

        np.savez(os.path.join(episode_path, "action.npz"), actions=actions)
