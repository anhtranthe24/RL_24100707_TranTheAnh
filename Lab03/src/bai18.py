import gymnasium as gym
from bai17 import first_visit_mc_prediction
from bai11 import stick_on_20_policy
env = gym.make("Blackjack-v1")
episode_counts = [100, 1000, 10000, 50000]
for n_episodes in episode_counts:
    V, return_count = first_visit_mc_prediction(
        env,
        stick_on_20_policy,
        n_episodes=n_episodes,
        gamma=1.0
    )
print(
    f"Episode: {n_episodes: >5} |" f"Estimated states: {len(V): >3}"
)
env.close()