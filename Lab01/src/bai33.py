def run_episode(env, policy, seed=None, max_steps=1000):
    obs, _ = env.reset(seed=seed)
    tot, steps, term, trunc = 0, 0, False, False
    for _ in range(max_steps):
        action = policy(obs) if policy else env.action_space.sample()
        obs, r, term, trunc, _ = env.step(action)
        tot += r
        steps += 1
        if term or trunc: break
    return {"reward": tot, "length": steps, "terminated": term, "truncated": trunc}