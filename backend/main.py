from rl.env import ElevatorEnv

env = ElevatorEnv()
state = env.reset()
print("Initial state:", state)

done = False
total_reward = 0
while not done:
    action = 0  # dummy policy for now — always pick elevator 0, just to test plumbing
    state, reward, done, _ = env.step(action)
    total_reward += reward

print("Episode finished. Total reward:", total_reward)
print("Completed requests:", env.building.completed_count)

total_wait = sum(
    req.passenger.waiting_time
    for req in env.building.completed_requests
)

completed_count = env.building.completed_count

if completed_count > 0:
    avg_wait = total_wait / completed_count
    print(f"Avg wait per passenger: {avg_wait:.1f} steps")
else:
    print("No completed passengers.")