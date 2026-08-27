import gymnasium as gym
env = gym.make("CartPole-v1")
env.reset(seed=42)
print("--- Bắt đầu mô phỏng tối đa 20 bước ---")
for t in range(20):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"t={t:2d}, action={action}, reward={reward}")
    if terminated or truncated:
        print(f"-> Episode kết thúc sớm tại bước {t}!")
        break
env.close()