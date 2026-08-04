from models.elevator import Elevator
from models.passenger import Passenger
from models.request import Request

elevator = Elevator(1)

passenger = Passenger(
    passenger_id=1,
    source_floor=2,
    destination_floor=8,
    arrival_time=0,
)

request = Request(
    request_id=1,
    passenger=passenger,
)

print(elevator)

# Dispatcher assigns request
elevator.assign_request(request)

print(elevator)

# Passenger boards
elevator.board_passenger(request)

print(elevator)

# Elevator moves
elevator.move_up()
elevator.move_up()

print(elevator)

# Passenger reaches destination
elevator.complete_request(request)

elevator.stop()

print(elevator)