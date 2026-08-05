"""
building.py

Defines the Building class that manages elevators,
requests, and the simulation environment.
"""

from collections import deque
from typing import Deque, List, Optional

from config import NUM_ELEVATORS
from models.elevator import Elevator
from models.request import Request
from utils.enums import ElevatorStatus


class Building:
    """
    Represents the entire building environment.
    """

    def __init__(self) -> None:

        # Create elevators
        self.elevators: List[Elevator] = [
            Elevator(elevator_id=i + 1)
            for i in range(NUM_ELEVATORS)
        ]

        # Waiting requests
        self.waiting_requests: Deque[Request] = deque()

        # Completed requests
        self.completed_requests: List[Request] = []

        # Simulation clock
        self.current_time: int = 0

    # -------------------------------------------------
    # Request Management
    # -------------------------------------------------

    def add_request(self, request: Request) -> None:
        """
        Add a new waiting request.
        """
        self.waiting_requests.append(request)

    def remove_request(self, request: Request) -> None:
        """
        Remove a waiting request.
        """
        if request in self.waiting_requests:
            self.waiting_requests.remove(request)

    def complete_request(self, request: Request) -> None:
        """
        Mark a request as completed.
        """
        request.complete()
        self.completed_requests.append(request)

    # -------------------------------------------------
    # Elevator Management
    # -------------------------------------------------

    def get_elevator(self, elevator_id: int) -> Optional[Elevator]:
        """
        Returns an elevator by ID.
        """

        for elevator in self.elevators:
            if elevator.elevator_id == elevator_id:
                return elevator

        return None

    def get_idle_elevators(self) -> List[Elevator]:
        """
        Returns all idle elevators.
        """

        return [
            elevator
            for elevator in self.elevators
            if elevator.status == ElevatorStatus.STOPPED
        ]

    def get_active_elevators(self) -> List[Elevator]:
        """
        Returns elevators currently moving.
        """

        return [
            elevator
            for elevator in self.elevators
            if elevator.status == ElevatorStatus.MOVING
        ]

    # -------------------------------------------------
    # Simulation Time
    # -------------------------------------------------

    def advance_time(self, steps: int = 1) -> None:
        """
        Advance simulation time.
        """

        self.current_time += steps

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    @property
    def total_requests(self) -> int:
        return (
            len(self.waiting_requests)
            + len(self.completed_requests)
            + sum(
                len(e.assigned_requests)
                + len(e.onboard_requests)
                for e in self.elevators
            )
        )

    @property
    def pending_requests(self) -> int:
        return len(self.waiting_requests)

    @property
    def completed_count(self) -> int:
        return len(self.completed_requests)

    def __repr__(self) -> str:

        return (
            f"Building("
            f"time={self.current_time}, "
            f"waiting={len(self.waiting_requests)}, "
            f"completed={len(self.completed_requests)}, "
            f"elevators={len(self.elevators)})"
        )