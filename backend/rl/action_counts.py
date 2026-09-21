from collections import Counter

from rl.env import ElevatorEnv
from rl.q_learning import QLearningAgent
from algorithms.nearest import NearestElevatorDispatcher
from algorithms.least_loaded import LeastLoadedDispatcher


def count_actions(env, pick):
    state = env.reset()
    counts = Counter()
    done = False
    while not done:
        action = pick(state)
        counts[action] += 1
        state, _, done, _ = env.step(action)
    return dict(sorted(counts.items()))


env = ElevatorEnv()

nearest = NearestElevatorDispatcher()
print("Nearest:      ", count_actions(env, lambda s: nearest.choose_action(env.building)))

least = LeastLoadedDispatcher()
print("Least loaded: ", count_actions(env, lambda s: least.choose_action(env.building)))

agent = QLearningAgent()
agent.load_q_table()
agent.epsilon = 0.0
print("Trained agent:", count_actions(env, agent.choose_action))