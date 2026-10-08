import gymnasium as gym
from collections import defaultdict
from mc_utils import generate_episode, compute_returns

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1

def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(lambda: defaultdict(float))
    returns_count = defaultdict(lambda: defaultdict(int))

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        rewards = [step[2] for step in episode]
        episode_returns = compute_returns(rewards, gamma)

        for (state, action, reward), G in zip(episode, episode_returns):
            returns_sum[state][action] += G
            returns_count[state][action] += 1

    Q = defaultdict(dict)

    for state in returns_sum:
        for action in returns_sum[state]:
            Q[state][action] = (
                returns_sum[state][action] / returns_count[state][action]
            )

    return Q

env = gym.make("Blackjack-v1")

Q = mc_action_value_prediction(
    env,
    stick_on_20_policy,
    n_episodes=10000,
    gamma=1.0
)

print("Number of states:", len(Q))

for state, action_values in list(Q.items())[:10]:
    print(f"State: {state} | Q: {dict(action_values)}")

env.close()

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    Q = mc_action_value_prediction(env, stick_on_20_policy, 10000)
    print("Number of states:", len(Q))
    for state, action_values in list(Q.items())[:10]:
        print(f"State: {state} | Q: {dict(action_values)}")
    env.close()