# evaluation/compare.py
# Runs every strategy, prints a comparison table, and saves a bar chart.
#
# Run from the backend/ folder with:   python -m evaluation.compare

import matplotlib.pyplot as plt

from rl.env import ElevatorEnv
from algorithms.nearest import NearestElevatorDispatcher
from algorithms.first_available import FirstAvailableDispatcher
from algorithms.round_robin import RoundRobinDispatcher
from algorithms.random_assign import RandomDispatcher
from evaluation.metrics import evaluate_many
from config import NUM_ELEVATORS

NUM_RUNS = 5  # episodes per strategy; results are averaged


def make_env():
    return ElevatorEnv()


def run_comparison():
    # name -> (strategy, is_agent)
    strategies = {
        "Nearest Elevator": (NearestElevatorDispatcher(), False),
        "First Available": (FirstAvailableDispatcher(), False),
        "Round Robin": (RoundRobinDispatcher(NUM_ELEVATORS), False),
        "Random": (RandomDispatcher(NUM_ELEVATORS), False),
        # Once Member 3 is done, add the trained agent like this:
        # "Q-Learning": (trained_agent, True),
    }

    results = {}
    print(f"{'Strategy':<20}{'Completed':<12}{'Avg Wait':<12}{'Total Reward':<15}")
    print("-" * 59)
    for name, (strategy, is_agent) in strategies.items():
        r = evaluate_many(make_env, strategy, is_agent, NUM_RUNS)
        results[name] = r
        print(f"{name:<20}{r['completed']:<12.1f}{r['avg_wait']:<12.1f}{r['total_reward']:<15.1f}")

    make_charts(results)
    return results


def make_charts(results):
    names = list(results.keys())

    # Chart 1: average waiting time (lower is better)
    plt.figure(figsize=(8, 5))
    plt.bar(names, [results[n]["avg_wait"] for n in names])
    plt.ylabel("Average waiting time")
    plt.title("Average passenger waiting time (lower is better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("avg_wait.png")

    # Chart 2: passengers served (higher is better)
    plt.figure(figsize=(8, 5))
    plt.bar(names, [results[n]["completed"] for n in names])
    plt.ylabel("Passengers served")
    plt.title("Passengers served (higher is better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("completed.png")

    print("\nSaved charts: avg_wait.png, completed.png")
    plt.show()


if __name__ == "__main__":
    run_comparison()