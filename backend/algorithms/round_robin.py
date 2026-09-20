from algorithms.base import BaseDispatcher

class RoundRobinDispatcher(BaseDispatcher):
    def __init__(self, num_elevators: int):
        self._next = 0
        self._num = num_elevators

    def choose_action(self, building) -> int:
        action = self._next
        self._next = (self._next + 1) % self._num
        return action