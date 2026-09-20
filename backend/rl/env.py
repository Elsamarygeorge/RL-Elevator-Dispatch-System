from environment.building import Building
from environment.traffic_generator import TrafficGenerator
from environment.simulator import Simulator
from config import NUM_ELEVATORS, RANDOM_SEED, SIMULATION_STEPS
from utils.enums import Direction, TrafficPeriod

DIR_CODE = {Direction.UP: 1, Direction.DOWN: -1, Direction.IDLE: 0}
PERIOD_CODE = {TrafficPeriod.MORNING: 0, TrafficPeriod.LUNCH: 1, TrafficPeriod.EVENING: 2}


class ElevatorEnv:
    """
    STATE:  for each elevator -> (floor, direction, load)
            + pending request -> (source_floor, destination_floor)
            + current period
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
            (e.current_floor, DIR_CODE[e.direction], e.current_load)
            for e in self.building.elevators
        )
        req = self.building.next_waiting_request()
        if req:
            req_state = (req.passenger.source_floor, req.passenger.destination_floor)
        else:
            req_state = (0, 0)  # no pending request right now
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