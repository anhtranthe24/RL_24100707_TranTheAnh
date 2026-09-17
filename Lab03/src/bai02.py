import gymnasium as gym
env=gym.make("Blackjack-v1")
for i in range(10):
    observation, info = env.reset()
    tongdiem, bai_dealer, ace_dung_duoc = observation
    print(f"\nEpisode: {i + 1}")
    print(f"Tổng điểm người chơi: {tongdiem}")
    print(f"Lá dealer: {bai_dealer}")
    print(f"Useable ace: {ace_dung_duoc}")
env.close()