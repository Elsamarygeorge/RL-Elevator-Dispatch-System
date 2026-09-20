from pathlib import Path

import matplotlib.pyplot as plt

from rl.env import ElevatorEnv
from rl.q_learning import QLearningAgent, train


# ----------------------------------------
# Output folder
# ----------------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# ----------------------------------------
# Train the Q-learning agent
# ----------------------------------------

env = ElevatorEnv()
agent = QLearningAgent()

history = train(
    env,
    agent,
    num_episodes=200
)

print("\nTraining completed.")
print("Episodes:", len(history))
print("States learned:", len(agent.q_table))


# ----------------------------------------
# Evaluate trained agent
# ----------------------------------------

# Turn off exploration during evaluation.
agent.epsilon = 0.0

state = env.reset()
done = False
total_reward = 0.0

while not done:
    action = agent.choose_action(state)

    state, reward, done, _ = env.step(action)

    total_reward += reward


completed = env.building.completed_count

total_wait = sum(
    r.passenger.waiting_time
    for r in env.building.completed_requests
)

avg_wait = (
    total_wait / completed
    if completed > 0
    else 0
)


# ----------------------------------------
# Print evaluation results
# ----------------------------------------

print("\nTrained Agent Evaluation")
print("------------------------")
print("Passengers served:", completed)
print("Passengers still waiting:", env.building.pending_requests)
print(f"Average waiting time: {avg_wait:.1f} steps")
print(f"Total reward: {total_reward:.1f}")


# ----------------------------------------
# Save results to a text file
# ----------------------------------------

results_file = results_dir / "q_learning_results.txt"

with open(results_file, "w", encoding="utf-8") as file:
    file.write("Q-LEARNING STAGE 4 RESULTS\n")
    file.write("==========================\n\n")

    file.write(f"Episodes: {len(history)}\n")
    file.write(f"States learned: {len(agent.q_table)}\n\n")

    file.write("Trained Agent Evaluation\n")
    file.write("------------------------\n")
    file.write(f"Passengers served: {completed}\n")
    file.write(
        f"Passengers still waiting: "
        f"{env.building.pending_requests}\n"
    )
    file.write(
        f"Average waiting time: "
        f"{avg_wait:.1f} steps\n"
    )
    file.write(
        f"Total reward: "
        f"{total_reward:.1f}\n"
    )

print("\nResults saved to:", results_file)


# ----------------------------------------
# Save reward curve as PNG
# ----------------------------------------

plot_file = results_dir / "q_learning_reward_curve.png"

plt.figure(figsize=(10, 5))
plt.plot(history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Reward")
plt.grid(True)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Reward curve saved to:", plot_file)