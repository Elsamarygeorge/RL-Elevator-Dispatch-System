from pathlib import Path
import matplotlib.pyplot as plt

from rl.env import ElevatorEnv
from rl.q_learning import QLearningAgent, train

# =====================================================
# Paths
# =====================================================
BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Configuration
# =====================================================
NUM_EPISODES = 1000
MOVING_AVERAGE_WINDOW = 10
EVAL_SEEDS = list(range(1000, 1010))  # held-out days used for evaluation
EVAL_EPISODES = len(EVAL_SEEDS)

# =====================================================
# Train (single fixed day)
# =====================================================
env = ElevatorEnv()
agent = QLearningAgent(epsilon_decay=0.99)

history = train(env, agent, num_episodes=NUM_EPISODES)
agent.save_q_table()

print("\nTraining completed.")
print("Episodes:", len(history))
print("States learned:", len(agent.q_table))
print("Final epsilon:", f"{agent.epsilon:.3f}")

# =====================================================
# Load into a fresh agent (checks the frontend can reuse it)
# =====================================================
loaded_agent = QLearningAgent()
loaded_agent.load_q_table()
print("States loaded:", len(loaded_agent.q_table))

# =====================================================
# Evaluate the trained agent on held-out days (no exploration)
# =====================================================
loaded_agent.epsilon = 0.0

served_list, waiting_list, avg_wait_list, reward_list = [], [], [], []

for eval_seed in EVAL_SEEDS:
    state = env.reset(seed=eval_seed)
    done = False
    total_reward = 0.0

    while not done:
        action = loaded_agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    completed = env.building.completed_count
    still_waiting = env.building.pending_requests
    total_wait = sum(
        r.passenger.waiting_time for r in env.building.completed_requests
    )
    avg_wait = total_wait / completed if completed > 0 else 0.0

    served_list.append(completed)
    waiting_list.append(still_waiting)
    avg_wait_list.append(avg_wait)
    reward_list.append(total_reward)


def mean(values):
    return sum(values) / len(values)


completed = mean(served_list)
still_waiting = mean(waiting_list)
avg_wait = mean(avg_wait_list)
total_reward = mean(reward_list)

print(f"\nTrained Agent Evaluation (mean of {EVAL_EPISODES} held-out days)")
print("------------------------")
print("Passengers served:", f"{completed:.1f}")
print("Passengers still waiting:", f"{still_waiting:.1f}")
print(f"Average waiting time: {avg_wait:.1f} steps")
print(f"Total reward: {total_reward:.1f}")

# =====================================================
# Save evaluation results
# =====================================================
results_file = RESULTS_DIR / "q_learning_results.txt"

with open(results_file, "w", encoding="utf-8") as file:
    file.write("Q-LEARNING STAGE 4 RESULTS\n")
    file.write("==========================\n\n")
    file.write(f"Training episodes: {len(history)}\n")
    file.write("Training traffic: single fixed day (default seed)\n")
    file.write(f"Evaluation traffic seeds: {EVAL_SEEDS[0]} to {EVAL_SEEDS[-1]}\n")
    file.write(f"Learning rate (alpha): {agent.alpha}\n")
    file.write(f"Discount factor (gamma): {agent.gamma}\n")
    file.write("Initial epsilon: 1.0\n")
    file.write(f"Epsilon decay: {agent.epsilon_decay}\n")
    file.write(f"Minimum epsilon: {agent.epsilon_min}\n")
    file.write(f"Final epsilon: {agent.epsilon:.3f}\n")
    file.write(f"States learned: {len(agent.q_table)}\n")
    file.write(f"States loaded: {len(loaded_agent.q_table)}\n\n")
    file.write(f"Trained Agent Evaluation (mean of {EVAL_EPISODES} held-out days)\n")
    file.write("------------------------\n")
    file.write(f"Passengers served: {completed:.1f}\n")
    file.write(f"Passengers still waiting: {still_waiting:.1f}\n")
    file.write(f"Average waiting time: {avg_wait:.1f} steps\n")
    file.write(f"Total reward: {total_reward:.1f}\n")

print("\nResults saved to:", results_file)

# =====================================================
# Moving average
# =====================================================
window = MOVING_AVERAGE_WINDOW

if len(history) >= window:
    moving_average = [
        sum(history[i - window + 1:i + 1]) / window
        for i in range(window - 1, len(history))
    ]
else:
    moving_average = []

# =====================================================
# Reward curve
# =====================================================
plot_file = RESULTS_DIR / "q_learning_reward_curve.png"

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(history) + 1), history, alpha=0.4, label="Raw reward")

if moving_average:
    plt.plot(
        range(window, len(history) + 1),
        moving_average,
        linewidth=2,
        label=f"{window}-episode moving average",
    )

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Reward")
plt.grid(True)
plt.legend()
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()

print("Reward curve saved to:", plot_file)