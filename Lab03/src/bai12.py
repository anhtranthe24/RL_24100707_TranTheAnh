import gymnasium as gym
from mc_utils import generate_episode
def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1
env = gym.make("Blackjack-v1")
n_episodes = 100
wins = 0
losses = 0
draws = 0
for _ in range(n_episodes):
    episode = generate_episode(env, stick_on_20_policy)
    final_reward = episode[-1][2]
    if final_reward > 0:
        wins += 1
    elif final_reward < 0:
        losses += 1
    else:
        draws += 1
print("Episodes:", n_episodes)
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
print(f"Win rate: {wins / n_episodes:.2%}")
print(f"Loss rate: {losses / n_episodes:.2%}")
print(f"Draw rate: {draws / n_episodes:.2%}")
env.close()