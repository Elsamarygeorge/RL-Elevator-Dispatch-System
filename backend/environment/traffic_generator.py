"""
traffic_generator.py

Generates passenger requests according to different
building traffic patterns.
"""

import random

from config import NUM_FLOORS
from environment.building import Building
from models.passenger import Passenger
from models.request import Request
from utils.helper import (
    next_passenger_id,
    next_request_id,
)


class TrafficGenerator:
    """
    Generates passengers based on
    different traffic conditions.
    """

    def __init__(self, seed: int | None = None):

        if seed is not None:
            random.seed(seed)

    # ----------------------------------------
    # Public Function
    # ----------------------------------------

    def generate_requests(
        self,
        building: Building,
        period: str,
        num_requests: int,
    ) -> None:
        """
        Generate requests and add them
        to the building.
        """

        for _ in range(num_requests):

            source, destination = self._generate_trip(period)

            passenger = Passenger(
                passenger_id=next_passenger_id(),
                source_floor=source,
                destination_floor=destination,
                arrival_time=building.current_time,
            )

            request = Request(
                request_id=next_request_id(),
                passenger=passenger,
            )

            building.add_request(request)

    # ----------------------------------------
    # Internal Helper
    # ----------------------------------------

    def _generate_trip(
        self,
        period: str,
    ) -> tuple[int, int]:

        period = period.lower()

        # Morning
        if period == "morning":

            source = 1
            destination = random.randint(2, NUM_FLOORS)

        # Evening
        elif period == "evening":

            source = random.randint(2, NUM_FLOORS)
            destination = 1

        # Lunch / Mixed

        else:

            source = random.randint(1, NUM_FLOORS)

            destination = source

            while destination == source:
                destination = random.randint(
                    1,
                    NUM_FLOORS,
                )

        return source, destination