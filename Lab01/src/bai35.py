import os
import matplotlib.pyplot as plt
from bai32 import improved_policy
from bai31 import angle_based_policy
from bai34 import evaluate_policy
if __name__ == "__main__":
    r_rand = evaluate_policy("CartPole-v1", None, 500)
    r_angle = evaluate_policy("CartPole-v1", angle_based_policy, 500)
    r_imp = evaluate_policy("CartPole-v1", improved_policy, 500)
    plt.plot(r_rand["all_rewards"], label='Random')
    plt.plot(r_angle["all_rewards"], label='Angle-based')
    plt.plot(r_imp["all_rewards"], label='Improved')
    plt.legend()
    plt.title("Comparison of 3 Agents")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "comparison_agents.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh thành công tại: {save_path}")
# NHẬN XÉT:
# 1. Random policy có hiệu suất kém nhất, biên độ dao động cực hẹp ở mức thấp.
# 2. Angle-based policy nhỉnh hơn, nhưng vẫn sụp đổ nhanh do bỏ qua gia tốc.
# 3. Improved policy kết hợp cả góc và vận tốc góc cho kết quả vượt trội,
# 4. cho thấy việc đọc hiểu chính xác observation mang tính quyết định trong RL.
# 5. Sự ổn định của thuật toán tỷ lệ thuận với lượng dữ liệu vật lý được xử lý.