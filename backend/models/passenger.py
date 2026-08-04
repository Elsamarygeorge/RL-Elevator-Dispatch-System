"""
passenger.py

Defines the Passenger class used in the elevator dispatch simulation.
"""

from __future__ import annotations

from typing import Optional

from config import NUM_FLOORS
from utils.enums import Direction


class Passenger:
    """
    Represents a passenger requesting elevator service.

    Attributes:
        passenger_id (int): Unique passenger ID.
        source_floor (int): Floor where the passenger starts.
        destination_floor (int): Floor the passenger wants to reach.
        arrival_time (int): Simulation step when the passenger appears.
    """

    def __init__(
        self,
        passenger_id: int,
        source_floor: int,
        destination_floor: int,
        arrival_time: int,
    ) -> None:

        # ---------- Validation ----------

        if passenger_id <= 0:
            raise ValueError("Passenger ID must be greater than 0.")

        if not (1 <= source_floor <= NUM_FLOORS):
            raise ValueError(
                f"Source floor must be between 1 and {NUM_FLOORS}."
            )

        if not (1 <= destination_floor <= NUM_FLOORS):
            raise ValueError(
                f"Destination floor must be between 1 and {NUM_FLOORS}."
            )

        if source_floor == destination_floor:
            raise ValueError(
                "Source and destination floors cannot be the same."
            )

        if arrival_time < 0:
            raise ValueError(
                "Arrival time cannot be negative."
            )

        # ---------- Passenger Information ----------

        self.passenger_id: int = passenger_id
        self.source_floor: int = source_floor
        self.destination_floor: int = destination_floor
        self.arrival_time: int = arrival_time

        # Updated during simulation
        self.boarding_time: Optional[int] = None
        self.exit_time: Optional[int] = None

        # Determine travel direction
        self.direction: Direction = (
            Direction.UP
            if destination_floor > source_floor
            else Direction.DOWN
        )

    @property
    def waiting_time(self) -> Optional[int]:
        """
        Returns the passenger's waiting time.
        """
        if self.boarding_time is None:
            return None

        return self.boarding_time - self.arrival_time

    @property
    def travel_time(self) -> Optional[int]:
        """
        Returns the passenger's travel time.
        """
        if self.boarding_time is None or self.exit_time is None:
            return None

        return self.exit_time - self.boarding_time

    @property
    def total_time(self) -> Optional[int]:
        """
        Returns the passenger's total journey time.
        """
        if self.exit_time is None:
            return None

        return self.exit_time - self.arrival_time

    def __repr__(self) -> str:
        return (
            f"Passenger("
            f"id={self.passenger_id}, "
            f"source={self.source_floor}, "
            f"destination={self.destination_floor}, "
            f"direction={self.direction.value})"
        )