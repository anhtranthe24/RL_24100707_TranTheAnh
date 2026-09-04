n_state = 2
n_actions = 2
P= [[
    [(1.0, 1, 1, False)],
    [(1.0, 0, 0, False)],
],
[
    [(1.0, 0, 2, False)],
    [(1.0, 1, 3, True)]
]]
print("Number of states (Số lượng trạng thái):", n_state)
print("Number of actions (Số lượng hành động):", n_actions)
print("\nMDP transition model (Mô hình chuyển tiếp MDP):")
for state in range(n_state):
    for action in range(n_actions):
        print(f"State {state}, Action {action}: ")
        for probability, next_state, reward, terminated in P[state][action]:
            print(
                f"  Probability = {probability}"
                f", Next state = {next_state}"
                f", Reward = {reward}"
                f", Terminated = {terminated}"
            )