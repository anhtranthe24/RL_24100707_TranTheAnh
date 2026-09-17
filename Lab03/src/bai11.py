import gymnasium as gym
def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1
env = gym.make("Blackjack-v1")
test_states = [
    (20, 10, False),
    (18, 6, False),
    (13, 2, False),
    (18, 6, True)
]
for state in test_states:
    action = stick_on_20_policy(state)
    action_name = "Stick" if action == 0 else "Hit"
    print(f"State: {state} -> Action: {action} ({action_name})")
env.close()