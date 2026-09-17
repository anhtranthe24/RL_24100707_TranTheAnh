import gymnasium as gym 
env=gym.make("Blackjack-v1")
trangthai, thongtin = env.reset()
terminated = False
truncated = False
while not terminated and not truncated:
    hanhdong = env.action_space.sample()
    trangthaimoi, phanthuong, terminated, truncated, thongtin = env.step(hanhdong)
    print("Trạng thái:", trangthai)
    print("Hành động:", hanhdong)
    print("Phần thưởng:", phanthuong)
    print("Trạng thái mới:", trangthaimoi)
    print("Terminated (Đã chấm dứt):", terminated)
    print("Truncated (Bị cắt):", truncated)
    print("-" * 40)
    trangthai = trangthaimoi
env.close()