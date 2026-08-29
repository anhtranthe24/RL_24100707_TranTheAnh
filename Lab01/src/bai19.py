import gymnasium as gym
import numpy as np
if __name__ == "__main__":
    observations = []
    for _ in range(10):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=42)
        observations.append(obs)
        env.close()
    is_identical = all(np.array_equal(observations[0], obs) for obs in observations) 
    print(f"Các observation khởi tạo có giống hệt nhau không? {is_identical}")
# Kết luận:
# Khi thiết lập cùng một giá trị random seed (ở đây là 42), môi trường sẽ luôn 
# sinh ra một trạng thái khởi tạo (initial observation) giống hệt nhau. 
# Cơ chế này đảm bảo tính toàn vẹn và khả năng tái lập chính xác của mọi thí nghiệm.