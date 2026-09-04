def print_frozenlake_policy(env, policy):
    symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    desc = env.unwrapped.desc
    n = desc.shape[0]
    for r in range(n):
        row = []
        for c in range(n):
            state = r * n + c
            cell = desc[r][c].decode()
            row.append(cell if cell in ["H", "G"] else symbols[policy[state]])
        print(" ".join(row))