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

        # Requests currently assigned to this elevator
        self.requests: List[Request] = []

    @property
    def current_load(self) -> int:
        """
        Returns the current number of assigned requests.
        """
        return len(self.requests)

    @property
    def available_capacity(self) -> int:
        """
        Returns remaining capacity.
        """
        return self.capacity - self.current_load

    def is_full(self) -> bool:
        """
        Checks whether the elevator is full.
        """
        return self.current_load >= self.capacity

    def add_request(self, request: Request) -> None:
        """
        Assigns a request to the elevator.
        """

        if self.is_full():
            raise ValueError(
                f"Elevator {self.elevator_id} is full."
            )

        self.requests.append(request)
        request.assign_elevator(self.elevator_id)

    def remove_request(self, request: Request) -> None:
        """
        Removes a completed request.
        """

        if request in self.requests:
            self.requests.remove(request)

    def move_up(self) -> None:
        """
        Moves the elevator up by one floor.
        """

        if self.current_floor < NUM_FLOORS:
            self.current_floor += 1
            self.direction = Direction.UP
            self.status = ElevatorStatus.MOVING

    def move_down(self) -> None:
        """
        Moves the elevator down by one floor.
        """

        if self.current_floor > 1:
            self.current_floor -= 1
            self.direction = Direction.DOWN
            self.status = ElevatorStatus.MOVING

    def stop(self) -> None:
        """
        Stops the elevator.
        """

        self.direction = Direction.IDLE
        self.status = ElevatorStatus.STOPPED

    def __repr__(self) -> str:
        return (
            f"Elevator("
            f"id={self.elevator_id}, "
            f"floor={self.current_floor}, "
            f"direction={self.direction.value}, "
            f"load={self.current_load}/{self.capacity})"
        )