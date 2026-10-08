import gymnasium as gym
from bai32 import on_policy_mc_control

def main():
    env = gym.make("Blackjack-v1")

    Q, policy, episode_rewards = on_policy_mc_control(
        env,
        n_episodes=100000,
        gamma=1.0,
        epsilon=0.1,
        seed=42
    )

    print("Training completed")
    print("Number of learned states:", len(Q))

    test_states = [
        (20, 10, False),
        (18, 6, False),
        (13, 2, False),
        (18, 6, True)
    ]

    for state in test_states:
        action = policy.get(state, None)

        if action is None:
            print(f"State: {state} | Not visited")
        else:
            action_name = "Stick" if action == 0 else "Hit"
            print(
                f"State: {state} | "
                f"Q: {Q[state]} | "
                f"Policy: {action} ({action_name})"
            )

    print("Total episodes:", len(episode_rewards))

    env.close()

if __name__ == "__main__":
    main()