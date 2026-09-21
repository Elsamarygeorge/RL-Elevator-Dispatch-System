from environment.building import Building
from environment.traffic_generator import TrafficGenerator
from environment.simulator import Simulator
from config import NUM_ELEVATORS, RANDOM_SEED, SIMULATION_STEPS
from utils.enums import Direction, TrafficPeriod

DIR_CODE = {Direction.UP: 1, Direction.DOWN: -1, Direction.IDLE: 0}
PERIOD_CODE = {TrafficPeriod.MORNING: 0, TrafficPeriod.LUNCH: 1, TrafficPeriod.EVENING: 2}

# --- State abstraction settings (tune these to trade detail vs. learnability) ---
ZONE_SIZE = 3             # floors per zone (v1 helpers, kept for compatibility)
LOAD_FULL_THRESHOLD = 6   # load at or above this counts as "full"; ~75% of elevator capacity

# --- State abstraction v2: features relative to the pending request ---
DIST_NEAR = 2   # up to this many floors away = "near"
DIST_MID = 5    # up to this many floors away = "mid", beyond = "far"

# --- Reward shaping weights (tunable) ---
ASSIGN_LOAD_COST = 1.0        # per request already held by the chosen elevator
ASSIGN_DIST_COST = 0.5        # per floor between the chosen elevator and the pickup floor
WAIT_COST_PER_REQUEST = 0.05  # per request in the system, per tick


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


def workload_bucket(n: int) -> int:
    """Requests already assigned to or carried by an elevator."""
    if n == 0:
        return 0
    if n <= 2:
        return 1
    if n <= 5:
        return 2
    return 3


def distance_bucket(d: int) -> int:
    """0 = near, 1 = mid, 2 = far (distance in floors to the pickup floor)."""
    if d <= DIST_NEAR:
        return 0
    if d <= DIST_MID:
        return 1
    return 2


class ElevatorEnv:
    """
    STATE:  for each elevator -> (workload bucket, distance bucket), where
            workload = requests assigned to it + passengers onboard
            (0, 1-2, 3-5, 6+) and distance = floors between the elevator
            and the pending request's pickup floor (near / mid / far).
            If no request is pending, the state is (-1,).
            At most 12^3 + 1 = 1729 states.
    ACTION: integer 0..NUM_ELEVATORS-1 - which elevator serves the
            current pending request
    REWARD (shaped):
            - at assignment: -(ASSIGN_LOAD_COST * requests the chosen elevator
              already holds + ASSIGN_DIST_COST * its distance to the pickup
              floor), so the consequence of the action is visible immediately
            - every tick: -WAIT_COST_PER_REQUEST for each request in the
              system (waiting, assigned or onboard)
            - -0.1 per elevator that moved with no pickup/drop-off work
    EPISODE: one full simulated day (SIMULATION_STEPS)
    """

    def __init__(self):
        self.building = None
        self.simulator = None

    def reset(self, seed=None):
        """Start a new simulated day. With no seed, use the default day
        (RANDOM_SEED); pass a seed to get a different traffic pattern."""
        self.building = Building()
        if seed is None:
            seed = RANDOM_SEED
        traffic_gen = TrafficGenerator(seed=seed)
        self.simulator = Simulator(self.building, traffic_gen)
        return self._encode_state()

    def _encode_state(self):
        req = self.building.next_waiting_request()
        if req is None:
            return (-1,)  # no pending request

        src = req.passenger.source_floor
        return tuple(
            (
                workload_bucket(len(e.assigned_requests) + len(e.onboard_requests)),
                distance_bucket(abs(e.current_floor - src)),
            )
            for e in self.building.elevators
        )

    def step(self, action: int):
        reward = 0.0
        req = self.building.next_waiting_request()

        if req is not None:
            elevator = self.building.elevators[action]

            # Immediate, action-dependent cost (measured before assigning)
            load_before = len(elevator.assigned_requests) + len(elevator.onboard_requests)
            dist = abs(elevator.current_floor - req.passenger.source_floor)
            reward -= ASSIGN_LOAD_COST * load_before + ASSIGN_DIST_COST * dist

            elevator.assign_request(req)
            self.building.remove_request(req)

        self.simulator.tick()

        # Small per-tick cost for every request still in the system
        pending = self.building.pending_requests
        if not isinstance(pending, int):
            pending = len(pending)
        in_system = pending + sum(
            len(e.assigned_requests) + len(e.onboard_requests)
            for e in self.building.elevators
        )
        reward -= WAIT_COST_PER_REQUEST * in_system

        # Small penalty for elevator movement with nothing to justify it
        for e in self.building.elevators:
            is_moving = e.status.value == "MOVING"
            has_work = bool(e.assigned_requests) or bool(e.onboard_requests)
            if is_moving and not has_work:
                reward -= 0.1

        done = self.building.current_time >= SIMULATION_STEPS
        next_state = self._encode_state()
        return next_state, reward, done, {}