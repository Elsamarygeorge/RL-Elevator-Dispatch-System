import random

from rl.env import ElevatorEnv
from rl.q_learning import QLearningAgent
from config import NUM_ELEVATORS

EPISODES = 10


def run_episode(env, pick_action):
    state = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        state, reward, done, _ = env.step(pick_action(state))
        total_reward += reward

    served = env.building.completed_count
    total_wait = sum(r.passenger.waiting_time for r in env.building.completed_requests)
    avg_wait = total_wait / served if served else 0.0
    return served, avg_wait, total_reward


def summarize(name, results):
    n = len(results)
    print(
        f"{name:<16} served={sum(r[0] for r in results)/n:.1f}  "
        f"avg_wait={sum(r[1] for r in results)/n:.1f}  "
        f"reward={sum(r[2] for r in results)/n:.1f}"
    )


# --- trained agent (greedy), tracking how often its state was seen in training ---
agent = QLearningAgent()
agent.load_q_table()
agent.epsilon = 0.0

hits = 0
steps = 0


def greedy(state):
    global hits, steps
    steps += 1
    if state in agent.q_table:      # check BEFORE choose_action creates the entry
        hits += 1
    return agent.choose_action(state)


env = ElevatorEnv()
agent_results = [run_episode(env, greedy) for _ in range(EPISODES)]

# --- pure random policy ---
rng = random.Random(0)
env = ElevatorEnv()
random_results = [
    run_episode(env, lambda s: rng.randint(0, NUM_ELEVATORS - 1))
    for _ in range(EPISODES)
]

print()
summarize("Trained agent", agent_results)
summarize("Random policy", random_results)
print(f"\nState hit rate during evaluation: {hits}/{steps} = {hits/steps:.1%}")