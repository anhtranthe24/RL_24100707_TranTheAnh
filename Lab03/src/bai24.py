import gymnasium as gym
from collections import defaultdict
from mc_utils import generate_episode, compute_returns

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1

env = gym.make("Blackjack-v1")
returns = defaultdict(list)
n_episodes = 10000

for _ in range(n_episodes):
    episode = generate_episode(env, stick_on_20_policy)
    rewards = [step[2] for step in episode]
    episode_returns = compute_returns(rewards, gamma=1.0)

    for (state, action, reward), G in zip(episode, episode_returns):
        returns[(state, action)].append(G)

print("Number of state-action pairs:", len(returns))

for (state, action), state_action_returns in list(returns.items())[:10]:
    print(
        f"State: {state} | "
        f"Action: {action} | "
        f"Returns: {state_action_returns[:10]} | "
        f"Count: {len(state_action_returns)}"
    )

env.close()