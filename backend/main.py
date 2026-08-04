from models.passenger import Passenger
from models.request import Request

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

print(request)

request.assign_elevator(2)

print(request)

request.complete()

print(request)