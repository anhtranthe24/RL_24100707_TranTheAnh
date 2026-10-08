import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from bai32 import on_policy_mc_control

def moving_average(values, window):
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return np.array([])

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )

def main():
    env = gym.make("Blackjack-v1")

    n_episodes = 100000
    window = 1000

    Q, policy, episode_rewards = on_policy_mc_control(
        env,
        n_episodes=n_episodes,
        gamma=1.0,
        epsilon=0.1,
        seed=42
    )

    rewards = np.asarray(episode_rewards, dtype=float)
    moving_avg = moving_average(rewards, window)

    print("Training completed")
    print("Total episodes:", len(rewards))
    print("Moving average window:", window)

    if len(moving_avg) > 0:
        print(f"Final moving average: {moving_avg[-1]:.4f}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.arange(window, n_episodes + 1),
        moving_avg
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Monte Carlo Control Learning Curve")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(
        figures_dir,
        "mc_convergence.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()

    print("Figure saved to:", output_path)

    env.close()

if __name__ == "__main__":
    main()