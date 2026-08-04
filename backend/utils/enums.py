"""
enums.py

Defines common enumerations used throughout the elevator dispatch system.
"""

from enum import Enum


class Direction(Enum):
    """Possible movement directions for an elevator or passenger."""

    UP = "UP"
    DOWN = "DOWN"
    IDLE = "IDLE"


class ElevatorStatus(Enum):
    """Current operating status of an elevator."""

    MOVING = "MOVING"
    STOPPED = "STOPPED"
    MAINTENANCE = "MAINTENANCE"


class RequestStatus(Enum):
    """Current status of a passenger request."""

    WAITING = "WAITING"
    ASSIGNED = "ASSIGNED"
    COMPLETED = "COMPLETED"


class TrafficPeriod(Enum):
    """Traffic periods simulated during the day."""

    MORNING = "Morning"
    LUNCH = "Lunch"
    EVENING = "Evening"