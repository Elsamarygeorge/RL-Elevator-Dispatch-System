import random

from rl.env import ElevatorEnv
from rl.q_learning import QLearningAgent
from algorithms.nearest import NearestElevatorDispatcher
from algorithms.least_loaded import LeastLoadedDispatcher
from config import NUM_ELEVATORS

# Fresh traffic days: not the training day, and not the 1000-1009 days
# used for evaluation in test_q_learning.py
SEEDS = list(range(2000, 2010))


def run_day(seed, pick):
    env = ElevatorEnv()
    state = env.reset(seed=seed)
    done = False
    while not done:
        action = pick(env, state)
        state, _, done, _ = env.step(action)
    served = env.building.completed_count
    total_wait = sum(r.passenger.waiting_time for r in env.building.completed_requests)
    return total_wait / served if served else 0.0


agent = QLearningAgent()
agent.load_q_table()
agent.epsilon = 0.0

nearest = NearestElevatorDispatcher()
least = LeastLoadedDispatcher()

results = {"Trained agent": [], "Random": [], "Nearest": [], "Least loaded": []}

for seed in SEEDS:
    rng = random.Random(seed)
    results["Trained agent"].append(run_day(seed, lambda env, s: agent.choose_action(s)))
    results["Random"].append(run_day(seed, lambda env, s: rng.randint(0, NUM_ELEVATORS - 1)))
    results["Nearest"].append(run_day(seed, lambda env, s: nearest.choose_action(env.building)))
    results["Least loaded"].append(run_day(seed, lambda env, s: least.choose_action(env.building)))

print(f"\nAvg wait on {len(SEEDS)} unseen traffic days (seeds {SEEDS[0]} to {SEEDS[-1]})")
for name, waits in results.items():
    print(f"{name:<14} mean={sum(waits)/len(waits):.1f}  min={min(waits):.1f}  max={max(waits):.1f}")

wins = sum(a < l for a, l in zip(results["Trained agent"], results["Least loaded"]))
print(f"\nAgent beat Least loaded on {wins}/{len(SEEDS)} days")