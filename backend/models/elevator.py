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

    def __init__(
        self,
        elevator_id: int,
        current_floor: int = 1,
    ) -> None:

        if elevator_id <= 0:
            raise ValueError("Elevator ID must be greater than 0.")

        if not (1 <= current_floor <= NUM_FLOORS):
            raise ValueError(
                f"Current floor must be between 1 and {NUM_FLOORS}."
            )

        self.elevator_id = elevator_id
        self.current_floor = current_floor

        self.direction = Direction.IDLE
        self.status = ElevatorStatus.STOPPED

        self.capacity = ELEVATOR_CAPACITY

        # Requests assigned but passenger not picked up yet
        self.assigned_requests: List[Request] = []

        # Requests whose passengers are inside elevator
        self.onboard_requests: List[Request] = []

    @property
    def current_load(self) -> int:
        return len(self.onboard_requests)

    @property
    def available_capacity(self) -> int:
        return self.capacity - self.current_load

    def is_full(self) -> bool:
        return self.current_load >= self.capacity

    def assign_request(self, request: Request) -> None:
        """
        Dispatcher assigns a request to this elevator.
        """

        self.assigned_requests.append(request)
        request.assign_elevator(self.elevator_id)

    def board_passenger(self, request: Request) -> None:
        """
        Passenger enters the elevator.
        """

        if self.is_full():
            raise ValueError(
                f"Elevator {self.elevator_id} is full."
            )

        if request in self.assigned_requests:
            self.assigned_requests.remove(request)

        self.onboard_requests.append(request)

    def complete_request(self, request: Request) -> None:
        """
        Passenger reaches destination.
        """

        if request in self.onboard_requests:
            self.onboard_requests.remove(request)

        request.complete()

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

    def __repr__(self):

        return (
            f"Elevator("
            f"id={self.elevator_id}, "
            f"floor={self.current_floor}, "
            f"direction={self.direction.value}, "
            f"assigned={len(self.assigned_requests)}, "
            f"onboard={self.current_load}/{self.capacity})"
        )