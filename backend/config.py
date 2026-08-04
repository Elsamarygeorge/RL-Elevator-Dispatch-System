"""
config.py

Stores configuration values used throughout the simulation.
"""

# ----------------------------
# Building Configuration
# ----------------------------

NUM_FLOORS = 10
NUM_ELEVATORS = 3

# ----------------------------
# Elevator Configuration
# ----------------------------

ELEVATOR_CAPACITY = 8
ELEVATOR_SPEED = 1      # Floors moved per simulation step

# ----------------------------
# Simulation Configuration
# ----------------------------

SIMULATION_STEPS = 1000

# Passenger generation probability
PEAK_REQUEST_RATE = 0.8
NORMAL_REQUEST_RATE = 0.5

# ----------------------------
# Random Seed
# ----------------------------

RANDOM_SEED = 42