import random
from config import PEAK_REQUEST_RATE, NORMAL_REQUEST_RATE
from utils.enums import TrafficPeriod


class Simulator:
    def __init__(self, building, traffic_generator):
        self.building = building
        self.traffic_generator = traffic_generator

    def _request_rate(self):
        return (
            NORMAL_REQUEST_RATE
            if self.building.current_period == TrafficPeriod.LUNCH
            else PEAK_REQUEST_RATE
        )

    def maybe_generate_traffic(self):
        if random.random() < self._request_rate():
            self.traffic_generator.generate_requests(
                self.building,
                self.building.current_period.value,
                num_requests=1,
            )

    def move_elevators(self):
        for elevator in self.building.elevators:
            target = elevator.next_stop_floor

            if target is None:
                elevator.stop()

            elif target > elevator.current_floor:
                elevator.move_up()

            elif target < elevator.current_floor:
                elevator.move_down()

            else:
                elevator.stop()
                self._resolve_arrivals(elevator)

    def _resolve_arrivals(self, elevator):
        # Board waiting passengers whose pickup floor == here
        for req in list(elevator.assigned_requests):
            if (
                req.passenger.source_floor == elevator.current_floor
                and not elevator.is_full()
            ):
                elevator.board_request(req)
                req.passenger.boarding_time = self.building.current_time

        # Drop off passengers whose destination == here
        for req in list(elevator.onboard_requests):
            if (
                req.passenger.destination_floor
                == elevator.current_floor
            ):
                elevator.remove_request(req)
                req.passenger.exit_time = self.building.current_time
                self.building.complete_request(req)

    def tick(self):
        """
        Advance the whole environment by one step.
        Does NOT assign new requests to elevators — that decision
        is an MDP action, made one layer up by Member 4's code.
        """
        self.maybe_generate_traffic()
        self.move_elevators()
        self.building.advance_time()