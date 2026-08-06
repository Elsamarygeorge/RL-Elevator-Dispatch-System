"""
elevator.py

Defines the Elevator class used in the elevator dispatch simulation.
"""

from __future__ import annotations

from typing import List

from config import ELEVATOR_CAPACITY, NUM_FLOORS
from models.request import Request
from utils.enums import Direction, ElevatorStatus


class Elevator:
    """
    Represents a single elevator in the building.
    """

    def __init__(
        self,
        elevator_id: int,
        current_floor: int = 1,
    ) -> None:

        # ---------- Validation ----------

        if elevator_id <= 0:
            raise ValueError("Elevator ID must be greater than 0.")

        if not (1 <= current_floor <= NUM_FLOORS):
            raise ValueError(
                f"Current floor must be between 1 and {NUM_FLOORS}."
            )

        # ---------- Elevator Information ----------

        self.elevator_id: int = elevator_id
        self.current_floor: int = current_floor

        self.direction: Direction = Direction.IDLE
        self.status: ElevatorStatus = ElevatorStatus.STOPPED

        self.capacity: int = ELEVATOR_CAPACITY

        # Requests assigned but passenger not picked up yet
        self.assigned_requests: List[Request] = []

        # Requests whose passengers are inside the elevator
        self.onboard_requests: List[Request] = []

    @property
    def current_load(self) -> int:
        """
        Returns the current number of passengers inside the elevator.
        """
        return len(self.onboard_requests)

    @property
    def available_capacity(self) -> int:
        """
        Returns the remaining elevator capacity.
        """
        return self.capacity - self.current_load
    
    @property
    def next_stop_floor(self) -> int | None:
        """
        Nearest floor this elevator still needs to visit —
        either to drop someone off, or (if there's room) pick
        someone up. When full, only drop-offs count, so the
        elevator is forced to unload before it can pick up more.
        """
        stops = [r.passenger.destination_floor for r in self.onboard_requests]
        if not self.is_full():
            stops += [r.passenger.source_floor for r in self.assigned_requests]
        if not stops:
            return None
        return min(stops, key=lambda f: abs(f - self.current_floor))
   

    def is_full(self) -> bool:
        """
        Checks whether the elevator is full.
        """
        return self.current_load >= self.capacity

    def assign_request(self, request: Request) -> None:
        """
        Assign a request to this elevator.
        """

        self.assigned_requests.append(request)
        request.assign_elevator(self)

    def board_request(self, request: Request) -> None:
        """
        Move an assigned request into the elevator.
        """

        if self.is_full():
            raise ValueError(
                f"Elevator {self.elevator_id} is full."
            )

        if request not in self.assigned_requests:
            raise ValueError(
                "Request is not assigned to this elevator."
            )

        self.assigned_requests.remove(request)
        self.onboard_requests.append(request)

    def remove_request(self, request: Request) -> None:
        """
        Remove a request after passenger exits.

        (Request completion is handled by the Building/Simulator.)
        """

        if request not in self.onboard_requests:
            raise ValueError(
                "Request is not inside the elevator."
            )

        self.onboard_requests.remove(request)

    def move_up(self) -> None:

        if self.current_floor < NUM_FLOORS:
            self.current_floor += 1
            self.direction = Direction.UP
            self.status = ElevatorStatus.MOVING

    def move_down(self) -> None:

        if self.current_floor > 1:
            self.current_floor -= 1
            self.direction = Direction.DOWN
            self.status = ElevatorStatus.MOVING

    def stop(self) -> None:

        self.direction = Direction.IDLE
        self.status = ElevatorStatus.STOPPED

    def __repr__(self) -> str:

        return (
            f"Elevator("
            f"id={self.elevator_id}, "
            f"floor={self.current_floor}, "
            f"direction={self.direction.value}, "
            f"assigned={len(self.assigned_requests)}, "
            f"onboard={self.current_load}/{self.capacity})"
        )