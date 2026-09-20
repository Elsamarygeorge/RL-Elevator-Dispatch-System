def evaluate_dispatcher(env, strategy, is_agent=False):
    """Run ONE full episode with a strategy and return summary metrics."""

    if is_agent:
        saved_epsilon = strategy.epsilon
        strategy.epsilon = 0.0

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        waiting_request = env.building.next_waiting_request()

        if is_agent:
            action = strategy.choose_action(state)
        elif waiting_request is not None:
            action = strategy.choose_action(env.building)
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_reward += reward

    if is_agent:
        strategy.epsilon = saved_epsilon

    completed = env.building.completed_count
    total_wait = sum(
        r.passenger.waiting_time
        for r in env.building.completed_requests
    )
    avg_wait = total_wait / completed if completed else 0

    return {
        "completed": completed,
        "still_waiting": env.building.pending_requests,
        "avg_wait": avg_wait,
        "total_reward": total_reward,
    }


def evaluate_many(make_env, strategy, is_agent=False, num_runs=5):
    """Run several episodes and return the average of each metric."""

    results = [
        evaluate_dispatcher(make_env(), strategy, is_agent)
        for _ in range(num_runs)
    ]

    return {
        key: sum(r[key] for r in results) / num_runs
        for key in results[0]
    }