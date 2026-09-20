import random
import pickle
from pathlib import Path
from collections import defaultdict

from config import NUM_ELEVATORS


class QLearningAgent:
    """
    Q-learning agent for elevator dispatch.

    The agent works only with:
        state
        action
        reward
        next_state

    It does not directly access Building, Elevator,
    Simulator, or other environment classes.
    """

    def __init__(
        self,
        num_actions=NUM_ELEVATORS,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        # Q-table:
        # state -> list of Q-values for each possible action
        self.q_table = defaultdict(
            lambda: [0.0] * num_actions
        )

        self.num_actions = num_actions

        # Learning rate
        self.alpha = alpha

        # Discount factor
        self.gamma = gamma

        # Exploration probability
        self.epsilon = epsilon

        # Exploration decay
        self.epsilon_decay = epsilon_decay

        # Minimum exploration probability
        self.epsilon_min = epsilon_min

        # Separate random generator for the agent.
        # This keeps agent exploration separate from
        # TrafficGenerator's random sequence.
        self.rng = random.Random(42)

    def choose_action(self, state) -> int:
        """
        Choose an action using epsilon-greedy policy.

        With probability epsilon:
            explore by choosing a random elevator.

        Otherwise:
            exploit by choosing the elevator with
            the highest Q-value for the current state.
        """

        if self.rng.random() < self.epsilon:
            return self.rng.randint(
                0,
                self.num_actions - 1
            )

        return max(
            range(self.num_actions),
            key=lambda a: self.q_table[state][a]
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> None:
        """
        Update the Q-value using the standard
        Q-learning update rule.
        """

        best_next = (
            0
            if done
            else max(self.q_table[next_state])
        )

        target = reward + self.gamma * best_next

        self.q_table[state][action] += (
            self.alpha
            * (
                target
                - self.q_table[state][action]
            )
        )

    def decay_epsilon(self) -> None:
        """
        Reduce exploration after each episode,
        while keeping epsilon above epsilon_min.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save_q_table(self, filepath=None) -> None:
        """
        Save the trained Q-table to a pickle file.

        By default, the file is saved as:
            backend/rl/q_table.pkl
        """

        if filepath is None:
            filepath = (
                Path(__file__).resolve().parent
                / "q_table.pkl"
            )

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Convert defaultdict to a normal dictionary
        # because the lambda inside defaultdict cannot
        # be directly pickled.
        with open(filepath, "wb") as file:
            pickle.dump(
                dict(self.q_table),
                file
            )

        print(f"Q-table saved to: {filepath}")

    def load_q_table(self, filepath=None) -> None:
        """
        Load a previously trained Q-table.

        By default, loads:
            backend/rl/q_table.pkl
        """

        if filepath is None:
            filepath = (
                Path(__file__).resolve().parent
                / "q_table.pkl"
            )

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Q-table file not found: {filepath}"
            )

        with open(filepath, "rb") as file:
            data = pickle.load(file)

        self.q_table.clear()
        self.q_table.update(data)

        print(f"Q-table loaded from: {filepath}")


def train(env, agent, num_episodes=200):
    """
    Train the Q-learning agent.

    Returns:
        history:
            Total reward obtained in each episode.
    """

    history = []

    for episode in range(num_episodes):

        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state
            total_reward += reward

        # Reduce exploration after each episode.
        agent.decay_epsilon()

        history.append(total_reward)

        if episode % 10 == 0:
            print(
                f"Episode {episode}: "
                f"reward={total_reward:.1f}, "
                f"epsilon={agent.epsilon:.3f}"
            )

    return history