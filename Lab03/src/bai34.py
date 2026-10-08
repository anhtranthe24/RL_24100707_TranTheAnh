import gymnasium as gym
import numpy as np
from bai32 import on_policy_mc_control

def random_policy(state, rng):
    return int(rng.integers(2))

def fixed_policy(state):
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0

    return 1

def learned_policy(state, policy):
    return policy.get(state, 0)

def evaluate_policy(env, policy, n_episodes=10000, seed=123):
    rng = np.random.default_rng(seed)

    wins = 0
    losses = 0
    draws = 0
    total_reward = 0

    for episode_number in range(n_episodes):
        state, info = env.reset(seed=seed + episode_number)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = policy(state, rng) if policy == random_policy else policy(state)

            state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_reward": total_reward / n_episodes
    }

def main():
    env = gym.make("Blackjack-v1")

    print("Training MC policy...")

    Q, learned_policy_map, episode_rewards = on_policy_mc_control(
        env,
        n_episodes=100000,
        gamma=1.0,
        epsilon=0.1,
        seed=42
    )

    random_result = evaluate_policy(
        env,
        random_policy,
        n_episodes=10000,
        seed=123
    )

    fixed_result = evaluate_policy(
        env,
        fixed_policy,
        n_episodes=10000,
        seed=123
    )

    def mc_policy(state):
        return learned_policy(state, learned_policy_map)

    learned_result = evaluate_policy(
        env,
        mc_policy,
        n_episodes=10000,
        seed=123
    )

    print("\nRandom Policy")
    print(f"Win rate: {random_result['win_rate']:.4f}")
    print(f"Loss rate: {random_result['loss_rate']:.4f}")
    print(f"Draw rate: {random_result['draw_rate']:.4f}")
    print(f"Mean reward: {random_result['mean_reward']:.4f}")

    print("\nFixed Policy")
    print(f"Win rate: {fixed_result['win_rate']:.4f}")
    print(f"Loss rate: {fixed_result['loss_rate']:.4f}")
    print(f"Draw rate: {fixed_result['draw_rate']:.4f}")
    print(f"Mean reward: {fixed_result['mean_reward']:.4f}")

    print("\nMC Learned Policy")
    print(f"Win rate: {learned_result['win_rate']:.4f}")
    print(f"Loss rate: {learned_result['loss_rate']:.4f}")
    print(f"Draw rate: {learned_result['draw_rate']:.4f}")
    print(f"Mean reward: {learned_result['mean_reward']:.4f}")

    env.close()

if __name__ == "__main__":
    main()