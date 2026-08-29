import gymnasium as gym
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env.action_space.seed(42)
    actions_run_1 = [env.action_space.sample() for _ in range(20)]
    env.action_space.seed(42)
    actions_run_2 = [env.action_space.sample() for _ in range(20)]
    print(f"Chuỗi action lần 1: {actions_run_1}")
    print(f"Chuỗi action lần 2: {actions_run_2}")
    is_identical = actions_run_1 == actions_run_2
    print(f"Hai chuỗi action có giống hệt nhau không? {is_identical}")
    env.close()