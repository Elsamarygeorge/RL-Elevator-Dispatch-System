# evaluation/compare.py
from rl.env import ElevatorEnv
from algorithms.nearest import NearestElevatorDispatcher
from algorithms.first_available import FirstAvailableDispatcher
from algorithms.round_robin import RoundRobinDispatcher
from algorithms.random_assign import RandomDispatcher
from evaluation.metrics import evaluate_dispatcher
from config import NUM_ELEVATORS

def run_comparison():
    strategies = {
        "Nearest Elevator": NearestElevatorDispatcher(),
        "First Available": FirstAvailableDispatcher(),
        "Round Robin": RoundRobinDispatcher(NUM_ELEVATORS),
        "Random": RandomDispatcher(NUM_ELEVATORS),
        # "Q-Learning": <load trained agent here once Member 3 is done>
    }

    print(f"{'Strategy':<20}{'Completed':<12}{'Avg Wait':<12}{'Total Reward':<15}")
    for name, dispatcher in strategies.items():
        env = ElevatorEnv()
        result = evaluate_dispatcher(env, dispatcher)
        print(f"{name:<20}{result['completed']:<12}{result['avg_wait']:<12.1f}{result['total_reward']:<15.1f}")

if __name__ == "__main__":
    run_comparison()