# evaluation/metrics.py
# Runs a strategy through the environment and measures how well it did.


def evaluate_dispatcher(env, strategy, is_agent=False):
    """Run ONE full episode with a strategy and return summary metrics.

    env       : an ElevatorEnv
    strategy  : a baseline dispatcher (Member 1) or the trained Q-learning agent (Member 3)
    is_agent  : True for the Q-learning agent, False for baselines
    """
    # For the trained agent we want it to always pick its BEST action,
    # not explore randomly. Epsilon = 0 turns exploration off.
    if is_agent:
        saved_epsilon = strategy.epsilon
        strategy.epsilon = 0.0

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if is_agent:
            action = strategy.choose_action(state)        # agent only sees the state
        else:
            action = strategy.choose_action(env.building)  # baselines look at the building
        state, reward, done, _ = env.step(action)
        total_reward += reward

    if is_agent:
        strategy.epsilon = saved_epsilon  # put it back so training isn't affected

    completed = env.building.completed_count
    total_wait = sum(r.passenger.waiting_time for r in env.building.completed_requests)
    avg_wait = total_wait / completed if completed else 0

    return {
        "completed": completed,
        "still_waiting": env.building.pending_requests,
        "avg_wait": avg_wait,
        "total_reward": total_reward,
    }


def evaluate_many(make_env, strategy, is_agent=False, num_runs=5):
    """Run several episodes and return the AVERAGE of each metric.

    One run can be lucky or unlucky (traffic is random), so averaging
    gives a fairer comparison, especially for the Random dispatcher.
    """
    results = [evaluate_dispatcher(make_env(), strategy, is_agent) for _ in range(num_runs)]
    return {key: sum(r[key] for r in results) / num_runs for key in results[0]}