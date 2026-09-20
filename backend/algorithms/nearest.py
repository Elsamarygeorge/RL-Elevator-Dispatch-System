from algorithms.base import BaseDispatcher

class NearestElevatorDispatcher(BaseDispatcher):
    def choose_action(self, building) -> int:
        req = building.next_waiting_request()
        distances = [
            abs(e.current_floor - req.passenger.source_floor)
            for e in building.elevators
        ]
        return distances.index(min(distances))