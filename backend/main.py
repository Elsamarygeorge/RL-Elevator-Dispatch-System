"""
main.py
Runs one full simulated day and prints a continuous,
readable trace of the environment in action.

Dispatch policy: NearestElevatorDispatcher (Stage 3 baseline).
Swap the `dispatcher = ...` line below to compare a different strategy.
"""

from rl.env import ElevatorEnv
from algorithms.nearest import NearestElevatorDispatcher
from config import NUM_ELEVATORS, SIMULATION_STEPS

# How often to print a full snapshot (every step is too noisy for a demo)
PRINT_EVERY = 25


def describe_elevator(idx, e):
    return f"E{idx}[floor={e.current_floor:2d}, load={e.current_load}/{e.capacity}]"


def main():
    print("=" * 70)

    env = ElevatorEnv()
    state = env.reset()

    # Dispatch strategy driving this run — swap this one line to compare
    # a different strategy (FirstAvailableDispatcher, RoundRobinDispatcher(NUM_ELEVATORS),
    # RandomDispatcher(NUM_ELEVATORS), or later the trained Q-learning agent).
    dispatcher = NearestElevatorDispatcher()

    print(f"\nSimulation reset. {NUM_ELEVATORS} elevators, all starting at floor 1.")
    print(f"Dispatch policy: {dispatcher.__class__.__name__}")
    print(f"Initial MDP state: {state}\n")

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        pending_req = env.building.next_waiting_request()
        action = dispatcher.choose_action(env.building)

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
                    f"Passenger request: floor {pending_req.passenger.source_floor} "
                    f"-> floor {pending_req.passenger.destination_floor}"
                )
                assign_label = f"Assigned -> E{action}"
            else:
                req_desc = "No new request this step"
                assign_label = "No assignment"

            print(
                f"[t={step:4d} | {period:>7s}] {req_desc:40s} | "
                f"{assign_label:16s} | {elevators_desc}"
            )

    # ---------------- Summary ----------------
    completed = env.building.completed_count
    total_wait = sum(
        req.passenger.waiting_time for req in env.building.completed_requests
    )
    avg_wait = total_wait / completed if completed > 0 else 0

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Dispatch policy            : {dispatcher.__class__.__name__}")
    print(f"Total simulated steps      : {SIMULATION_STEPS}")
    print(f"Passengers served          : {completed}")
    print(f"Passengers still waiting   : {env.building.pending_requests}")
    print(f"Average waiting time       : {avg_wait:.1f} steps")
    print(f"Total reward (policy score): {total_reward:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()