import gymnasium as gym
import numpy as np
def test_frozenlake(is_slippery, episodes=500):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    success, lengths, rewards = 0, [], []
    for _ in range(episodes):
        env.reset()
        steps, tot = 0, 0
        while True:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            tot += r
            steps += 1
            if term or trunc:
                if r == 1.0: success += 1
                break
        lengths.append(steps)
        rewards.append(tot)
    env.close()
    return success / episodes, np.mean(rewards), np.mean(lengths)
if __name__ == "__main__":
    sr_f, mr_f, ml_f = test_frozenlake(False)
    sr_t, mr_t, ml_t = test_frozenlake(True)
    print(f"Deterministic (False): SR={sr_f}, Mean Reward={mr_f}, Mean Length={ml_f}")
    print(f"Stochastic (True)  : SR={sr_t}, Mean Reward={mr_t}, Mean Length={ml_t}")
# KẾT LUẬN:
# Môi trường Deterministic (is_slippery=False) dễ đoán hơn, action hoạt động chính xác.
# Môi trường Stochastic (is_slippery=True) trơn trượt, action bị nhiễu ngẫu nhiên
# dẫn đến success rate và average reward giảm mạnh dù dùng cùng thuật toán random.