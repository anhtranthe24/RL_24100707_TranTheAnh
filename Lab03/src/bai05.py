import gymnasium as gym
import numpy as np
from mc_utils import generate_episode
def random_policy(trang_thai):
    return np.random.randint(0, 2)
env = gym.make("Blackjack-v1")
episode = generate_episode(env, random_policy)
print("Độ dài episode:", len(episode))
for i, (trang_thai, hanh_dong, phan_thuong) in enumerate(episode):
    print(f"Bước {i + 1}:")
    print("  Trạng thái:", trang_thai)
    print("  Hành động:", hanh_dong)
    print("  Phần thưởng:", phan_thuong)
env.close()