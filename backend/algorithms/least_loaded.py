from algorithms.base import BaseDispatcher


class LeastLoadedDispatcher(BaseDispatcher):
    """Pick the elevator with the fewest assigned+onboard requests;
    break ties by distance to the pickup floor, then by index."""

    def choose_action(self, building) -> int:
        req = building.next_waiting_request()
        if req is None:
            return 0
        src = req.passenger.source_floor

        def key(i):
            e = building.elevators[i]
            load = len(e.assigned_requests) + len(e.onboard_requests)
            return (load, abs(e.current_floor - src), i)

        return min(range(len(building.elevators)), key=key)