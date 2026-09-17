def generate_episode(env, policy, seed=None):
    trang_thai, thong_tin = env.reset(seed=seed)
    episode = []
    terminated = False
    truncated = False
    while not terminated and not truncated:
        hanh_dong = policy(trang_thai)
        trang_thai_moi, phan_thuong, terminated, truncated, thong_tin = env.step(hanh_dong)
        episode.append((trang_thai, hanh_dong, phan_thuong))
        trang_thai = trang_thai_moi
    return episode
def compute_returns(rewards, gamma=1.0):
    returns = [0.0] * len(rewards)
    G= 0.0
    for t in range(len(rewards) -1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns