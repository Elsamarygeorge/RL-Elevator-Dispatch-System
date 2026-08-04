from models.passenger import Passenger

p = Passenger(
    passenger_id=1,
    source_floor=2,
    destination_floor=8,
    arrival_time=0
)

print(p)

print("Waiting:", p.waiting_time)

p.boarding_time = 5

print("Waiting:", p.waiting_time)

p.exit_time = 12

print("Travel:", p.travel_time)
print("Total :", p.total_time)