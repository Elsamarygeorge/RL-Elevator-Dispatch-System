from algorithms.base import BaseDispatcher

class NearestElevatorDispatcher(BaseDispatcher):
    def choose_action(self, building) -> int:
        req = building.next_waiting_request()
        if req is None:
            # No pending request right now — the returned action is ignored
            # by ElevatorEnv.step() in this case anyway, so any value is safe.
            return 0
        distances = [
            abs(e.current_floor - req.passenger.source_floor)
            for e in building.elevators
        ]
        return distances.index(min(distances))