import gymnasium as gym
env = gym.make("CartPole-v1")
observation, info = env.reset()
for t in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
# terminated có ý nghĩa gì?
# -> Báo hiệu agent đã đi đến trạng thái kết thúc tự nhiên của bài toán (ví dụ: cột bị đổ, xe trượt khỏi ranh giới).
# truncated có ý nghĩa gì?
# -> Báo hiệu episode bị ngắt quãng cưỡng bức bởi yếu tố bên ngoài (thường là do chạm mức tối đa 500 steps).
# Vì sao không nên dùng done của API cũ?
# -> 'done' gộp chung terminated và truncated, khiến Agent không phân biệt được nó thất bại do năng lực kém hay bị hệ thống ép dừng. Việc này gây sai lệch nghiêm trọng đến hàm tính toán Reward trong các thuật toán RL sâu hơn.