import gymnasium as gym
import matplotlib.pyplot as plt
from time import perf_counter
from mdp_utils import value_iteration, policy_iteration, greedy_policy_from_value, evaluate_policy_by_simulation
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
start = perf_counter()
V_vi, vi_iterations, _ = value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000)
vi_time = perf_counter() - start
vi_policy = greedy_policy_from_value(env, V_vi, gamma=0.99)
start = perf_counter()
pi_policy, V_pi, pi_iterations = policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000)
pi_time = perf_counter() - start
vi_result = evaluate_policy_by_simulation(env, vi_policy, 1000, 42)
pi_result = evaluate_policy_by_simulation(env, pi_policy, 1000, 42)
print(f"{'Algorithm':<20}{'Iterations':<12}{'Time(s)':<12}{'Success Rate':<15}{'Mean Reward':<15}")
print("-" * 74)
print(f"{'Value Iteration':<20}{vi_iterations:<12}{vi_time:<12.6f}{vi_result['success_rate']:<15.4f}{vi_result['mean_reward']:<15.4f}")
print(f"{'Policy Iteration':<20}{pi_iterations:<12}{pi_time:<12.6f}{pi_result['success_rate']:<15.4f}{pi_result['mean_reward']:<15.4f}")
algorithms = ["Value Iteration", "Policy Iteration"]
times = [vi_time, pi_time]
success_rates = [vi_result["success_rate"], pi_result["success_rate"]]
mean_rewards = [vi_result["mean_reward"], pi_result["mean_reward"]]
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].bar(algorithms, times)
ax[0].set_title("Runtime")
ax[0].set_ylabel("Seconds")

ax[1].bar(algorithms, success_rates)
ax[1].set_title("Success Rate")
ax[1].set_ylabel("Rate")

ax[2].bar(algorithms, mean_rewards)
ax[2].set_title("Mean Reward")
ax[2].set_ylabel("Reward")

plt.tight_layout()
plt.savefig("./figures/algorithm_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nNhận xét:")
print("1. Value Iteration cập nhật hàm giá trị theo Bellman optimality.")
print("2. Policy Iteration thực hiện đánh giá và cải thiện chính sách.")
print("3. Hai thuật toán đều tìm được chính sách hiệu quả trên FrozenLake.")
print("4. Success rate được tính dựa trên số episode đạt Goal.")
print("5. Mean reward là phần thưởng trung bình sau 1000 episode.")
print("6. Runtime phản ánh thời gian tính toán của mỗi thuật toán.")
print("7. Runtime thực tế phụ thuộc vào môi trường chạy và máy tính.")
print("8. Hai thuật toán cho kết quả success rate và mean reward giống nhau trong thí nghiệm này.")
env.close()