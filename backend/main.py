from models.elevator import Elevator
from models.passenger import Passenger
from models.request import Request
from utils.helper import next_passenger_id, next_request_id, floor_distance

pid = next_passenger_id()
rid = next_request_id()

passenger = Passenger(
    passenger_id=pid,
    source_floor=2,
    destination_floor=8,
    arrival_time=0,
)

request = Request(
    request_id=rid,
    passenger=passenger,
)

elevator = Elevator(1)

elevator.assign_request(request)

print(passenger)
print(request)
print(elevator)

print("Distance:", floor_distance(2, 8))