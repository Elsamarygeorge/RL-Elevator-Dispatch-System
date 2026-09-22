"""
main.py
Runs one full simulated day and prints a continuous,
readable trace of the environment in action — including
the reward for every step, so it's visible proof of the
MDP loop (State -> Action -> Reward), not just a final total.

Dispatch policy: the trained Q-learning agent (Stage 4 result).
"""

from rl.env import ElevatorEnv
from rl.q_learning import QLearningAgent
from config import NUM_ELEVATORS, SIMULATION_STEPS

PRINT_EVERY = 25


class QLearningDispatcher:
    """
    Adapts the trained QLearningAgent (choose_action(state)) to the same
    interface every other dispatcher uses (choose_action(building)), by
    asking the bound ElevatorEnv for its own current encoded state.
    """
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.__class__.__name__ = "QLearningDispatcher"

    def choose_action(self, building):
        state = self.env._encode_state()
        return self.agent.choose_action(state)


def describe_elevator(idx, e):
    return f"E{idx}[floor={e.current_floor:2d}, load={e.current_load}/{e.capacity}]"


def choose_action_safely(dispatcher, building):
    """
    Only asks the dispatcher for an action when there's an actual pending
    request to decide on. Required for correctness with stateful
    dispatchers like RoundRobinDispatcher — calling choose_action() on
    every tick (even no-op ticks) would silently advance its internal
    counter on wasted calls, desyncing its real assignment sequence.
    """
    if building.next_waiting_request() is None:
        return 0
    return dispatcher.choose_action(building)


def total_waiting(building):
    """Total passengers currently waiting anywhere in the building —
    includes both unassigned requests and requests already assigned to
    an elevator but not yet boarded."""
    count = len(building.waiting_requests)
    for e in building.elevators:
        count += len(e.assigned_requests)
    return count


def main():
    print("=" * 90)

    env = ElevatorEnv()
    import random
    state = env.reset(seed=random.randint(0, 2**31 - 1))

    agent = QLearningAgent()
    agent.load_q_table()
    agent.epsilon = 0.0  # greedy — no exploration during a demo run
    dispatcher = QLearningDispatcher(env, agent)

    print(f"\nSimulation reset. {NUM_ELEVATORS} elevators, all starting at floor 1.")
    print(f"Dispatch policy: {dispatcher.__class__.__name__}")
    print(f"Initial MDP state: {state}\n")

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        pending_req = env.building.next_waiting_request()
        action = choose_action_safely(dispatcher, env.building)

        state, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

        if step % PRINT_EVERY == 0 or pending_req is not None:
            period = env.building.current_period.value
            elevators_desc = " | ".join(
                describe_elevator(i, e) for i, e in enumerate(env.building.elevators)
            )

            if pending_req is not None:
                req_desc = (
                    f"Passenger request: floor {pending_req.passenger.source_floor}"
                    f"->{pending_req.passenger.destination_floor}"
                )
                assign_label = f"Assigned-> E{action}"
            else:
                req_desc = "No new request this step"
                assign_label = "No assignment"

            print(
                f"[t={step:4d} | {period:>7s}] {req_desc:30s} | "
                f"{assign_label:14s} | Reward:{reward:6.2f} | {elevators_desc}"
            )

    # ---------------- Summary ----------------
    completed = env.building.completed_count
    total_wait = sum(
        req.passenger.waiting_time for req in env.building.completed_requests
    )
    avg_wait = total_wait / completed if completed > 0 else 0

    print("\n" + "=" * 90)
    print("SIMULATION COMPLETE")
    print("=" * 90)
    print(f"Dispatch policy            : {dispatcher.__class__.__name__}")
    print(f"Total simulated steps      : {SIMULATION_STEPS}")
    print(f"Passengers still waiting   : {total_waiting(env.building)}")
    print(f"Passengers served          : {completed}")
    print(f"Average waiting time       : {avg_wait:.1f} steps")
    print(f"Total reward (policy score): {total_reward:.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()