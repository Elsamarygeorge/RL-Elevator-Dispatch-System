from environment.building import Building
from environment.traffic_generator import TrafficGenerator
from environment.simulator import Simulator
from config import NUM_ELEVATORS, RANDOM_SEED, SIMULATION_STEPS
from utils.enums import Direction, TrafficPeriod

DIR_CODE = {Direction.UP: 1, Direction.DOWN: -1, Direction.IDLE: 0}
PERIOD_CODE = {TrafficPeriod.MORNING: 0, TrafficPeriod.LUNCH: 1, TrafficPeriod.EVENING: 2}

# --- State abstraction settings (tune these to trade detail vs. learnability) ---
ZONE_SIZE = 3             # floors per zone; 3 -> ~4 zones for 10 floors, 2 -> 5 zones (finer)
LOAD_FULL_THRESHOLD = 6   # load at or above this counts as "full"; set to ~75% of elevator capacity


def floor_to_zone(floor: int) -> int:
    """Group floors into zones. Works whether floors start at 0 or 1."""
    return floor // ZONE_SIZE


def load_bucket(load: int) -> int:
    """0 = empty, 1 = partial, 2 = (nearly) full."""
    if load == 0:
        return 0
    if load < LOAD_FULL_THRESHOLD:
        return 1
    return 2


class ElevatorEnv:
    """
    STATE:  for each elevator -> (floor zone, direction, load bucket)
            + pending request -> (source floor zone, direction of travel)
            + current period
            (abstracted on purpose: exact floors/loads gave ~87,000 states,
            far too many for tabular Q-learning to cover)
    ACTION: integer 0..NUM_ELEVATORS-1 — which elevator serves the
            current pending request
    REWARD: -1 * waiting_time of each request completed this step
            -0.1 per elevator that moved with no pickup/drop-off
            work assigned (penalizes truly unnecessary movement,
            not movement in general)
    EPISODE: one full simulated day (SIMULATION_STEPS)
    """

    def __init__(self):
        self.building = None
        self.simulator = None

    def reset(self):
        self.building = Building()
        traffic_gen = TrafficGenerator(seed=RANDOM_SEED)
        self.simulator = Simulator(self.building, traffic_gen)
        return self._encode_state()

    def _encode_state(self):
        elevator_state = tuple(
            (floor_to_zone(e.current_floor), DIR_CODE[e.direction], load_bucket(e.current_load))
            for e in self.building.elevators
        )
        req = self.building.next_waiting_request()
        if req:
            src = req.passenger.source_floor
            dst = req.passenger.destination_floor
            travel_dir = 1 if dst > src else (-1 if dst < src else 0)
            req_state = (floor_to_zone(src), travel_dir)
        else:
            req_state = (-1, 0)  # no pending request (-1 can't clash with a real zone)
        period_state = PERIOD_CODE[self.building.current_period]
        return elevator_state + req_state + (period_state,)

    def step(self, action: int):
        reward = 0.0
        req = self.building.next_waiting_request()

        if req is not None:
            elevator = self.building.elevators[action]
            elevator.assign_request(req)
            self.building.remove_request(req)

        completed_before = self.building.completed_count
        self.simulator.tick()
        completed_after = self.building.completed_count

        # Reward: penalize waiting time for every request completed this step
        newly_completed = self.building.completed_requests[completed_before:completed_after]
        for r in newly_completed:
            reward -= r.passenger.waiting_time or 0

        # Small penalty for elevator movement with nothing to justify it
        for e in self.building.elevators:
            is_moving = e.status.value == "MOVING"
            has_work = bool(e.assigned_requests) or bool(e.onboard_requests)
            if is_moving and not has_work:
                reward -= 0.1

        done = self.building.current_time >= SIMULATION_STEPS
        next_state = self._encode_state()
        return next_state, reward, done, {}