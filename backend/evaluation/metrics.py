# evaluation/metrics.py
def evaluate_dispatcher(env, dispatcher_or_agent, is_agent=False):
    """Run one full episode with a given strategy and return summary metrics."""
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = (
            dispatcher_or_agent.choose_action(state) if is_agent
            else dispatcher_or_agent.choose_action(env.building)
        )
        state, reward, done, _ = env.step(action)
        total_reward += reward

    completed = env.building.completed_count
    total_wait = sum(r.passenger.waiting_time for r in env.building.completed_requests)
    avg_wait = total_wait / completed if completed else 0

    return {
        "completed": completed,
        "still_waiting": env.building.pending_requests,
        "avg_wait": avg_wait,
        "total_reward": total_reward,
    }