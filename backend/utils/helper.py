"""
helper.py

Utility functions used throughout the elevator dispatch simulation.
"""

from itertools import count

# Auto-increment counters
_passenger_counter = count(1)
_request_counter = count(1)


def next_passenger_id() -> int:
    """
    Returns the next passenger ID.
    """
    return next(_passenger_counter)


def next_request_id() -> int:
    """
    Returns the next request ID.
    """
    return next(_request_counter)


def floor_distance(floor_a: int, floor_b: int) -> int:
    """
    Returns the distance between two floors.
    """
    return abs(floor_a - floor_b)