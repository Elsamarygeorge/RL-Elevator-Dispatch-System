"""
inspect_q_table.py
Loads the trained Q-table and prints it in a human-readable form —
for showing your teacher what the agent actually learned, since the
raw .pkl file is binary and can't be opened directly.

Run from the backend/ folder:
    python -m rl.inspect_q_table
"""

from rl.q_learning import QLearningAgent


def main():
    agent = QLearningAgent()
    agent.load_q_table()

    print("=" * 70)
    print("TRAINED Q-TABLE SUMMARY")
    print("=" * 70)
    print(f"Total distinct states learned: {len(agent.q_table)}")
    print()

    print("Example states and what the agent learned to do in each:")
    print("-" * 70)

    # Show a handful of example states — enough to demonstrate the
    # structure without dumping all ~1700 of them.
    example_count = 10
    print(f"{'State':<45} {'Q(E0)':>8} {'Q(E1)':>8} {'Q(E2)':>8}  Best")
    print("-" * 80)

    for i, (state, q_values) in enumerate(agent.q_table.items()):
        if i >= example_count:
            break
        best_action = max(range(len(q_values)), key=lambda a: q_values[a])
        print(f"{str(state):<45} {q_values[0]:>8.2f} {q_values[1]:>8.2f} {q_values[2]:>8.2f}  E{best_action}")

    print("-" * 70)
    print(f"(Showing {example_count} of {len(agent.q_table)} total learned states)")
    print("=" * 70)


if __name__ == "__main__":
    main()