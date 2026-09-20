import random
from algorithms.base import BaseDispatcher

class RandomDispatcher(BaseDispatcher):
    def __init__(self, num_elevators: int):
        self._num = num_elevators

    def choose_action(self, building) -> int:
        return random.randint(0, self._num - 1)