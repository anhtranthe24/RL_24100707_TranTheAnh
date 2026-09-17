import gymnasium as gym
env=gym.make("Blackjack-v1")
ACTION_NAMES={
    0: "stick",
    1: "Hit"
}
print("Không gian hành động:", env.action_space)
for hanhdong, ten in ACTION_NAMES.items():
    print(f"{hanhdong}: {ten}")
env.close()