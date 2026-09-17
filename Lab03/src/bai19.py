import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from bai17 import first_visit_mc_prediction
from bai11 import stick_on_20_policy
env = gym.make("Blackjack-v1")
V, returns_count = first_visit_mc_prediction(
    env,
    stick_on_20_policy,
    n_episodes=50000,
    gamma=1.0
)
player_sums = range(12, 22)
dealer_cards = range(1, 11)
value_table = np.full((len(player_sums), len(dealer_cards)), np.nan)
for i, player_sum in enumerate(player_sums):
    for j, dealer_card in enumerate(dealer_cards):
        state = (player_sum, dealer_card, False)
        if state in V:
            value_table[i, j] = V[state]
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
plt.figure(figsize=(10, 6))
plt.imshow(value_table, aspect="auto", origin="lower")
plt.colorbar(label="V(s)")
plt.xticks(range(10), dealer_cards)
plt.yticks(range(10), player_sums)
plt.xlabel("Dealer Showing Card")
plt.ylabel("Player Sum")
plt.title("First-Visit Monte Carlo Value Function")
plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "first_visit_value.png"),
    dpi=300
)
plt.show()
env.close()