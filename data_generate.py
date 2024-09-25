import gymnasium as gym
from stable_baselines3 import PPO
import cv2
import os
import numpy as np
import flappy_bird_gymnasium 

# 创建 Flappy Bird 环境
env = gym.make("Flappy2Birds-v0", render_mode="rgb_array")

# 使用 PPO 算法并选择 MLP 策略
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

# 设置要运行的回合数
num_episodes = 10  
for episode in range(1, num_episodes + 1):
    print(f"episode:{episode:04d}")  # 使用四位数格式化
    obs, _ = env.reset()
    
    # 格式化 episode 路径
    episode_path = f"data_flappy_bird_2/episode{episode:04d}"
    os.makedirs(episode_path, exist_ok=True)
    
    actions = []
    t = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        # 获取当前帧的图像并保存为 0001.png, 0002.png, ...
        img = env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式以适用于 OpenCV
        cv2.imwrite(os.path.join(episode_path, f"{t + 1:03d}.png"), img)

        actions.append(action)
        t += 1

    # 保存动作数据
    np.savez_compressed(os.path.join(episode_path, "action.npz"), actions=actions)
