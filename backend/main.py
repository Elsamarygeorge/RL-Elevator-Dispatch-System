from environment.building import Building
from environment.traffic_generator import TrafficGenerator

building = Building()

generator = TrafficGenerator(seed=42)

generator.generate_requests(
    building=building,
    period="morning",
    num_requests=5,
)

print(building)

print("\nWaiting Requests:\n")

for request in building.waiting_requests:
    print(request)