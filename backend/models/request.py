"""
request.py

Defines the Request class used in the elevator dispatch simulation.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from models.passenger import Passenger
from utils.enums import RequestStatus

if TYPE_CHECKING:
    from models.elevator import Elevator


class Request:
    """
    Represents a passenger's elevator request.
    """

    def __init__(
        self,
        request_id: int,
        passenger: Passenger,
    ) -> None:

        if request_id <= 0:
            raise ValueError(
                "Request ID must be greater than 0."
            )

        self.request_id: int = request_id
        self.passenger: Passenger = passenger

        # Assigned during dispatching
        self.assigned_elevator: Optional["Elevator"] = None

        self.status: RequestStatus = RequestStatus.WAITING

    def assign_elevator(self, elevator: "Elevator") -> None:
        """
        Assigns the request to an elevator.
        """

        self.assigned_elevator = elevator
        self.status = RequestStatus.ASSIGNED

    def complete(self) -> None:
        """
        Marks the request as completed.
        """

        self.status = RequestStatus.COMPLETED

    def __repr__(self) -> str:

        elevator_id = (
            self.assigned_elevator.elevator_id
            if self.assigned_elevator
            else None
        )

        return (
            f"Request("
            f"id={self.request_id}, "
            f"passenger={self.passenger.passenger_id}, "
            f"status={self.status.value}, "
            f"assigned_elevator={elevator_id})"
        )