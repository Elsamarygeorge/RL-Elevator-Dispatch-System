from algorithms.base import BaseDispatcher
from utils.enums import ElevatorStatus

class FirstAvailableDispatcher(BaseDispatcher):
    def choose_action(self, building) -> int:
        for i, e in enumerate(building.elevators):
            if e.status == ElevatorStatus.STOPPED:
                return i
        return 0