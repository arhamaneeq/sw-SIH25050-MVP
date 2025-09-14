import os
import traci
import matplotlib.pyplot as plt

# Ensure SUMO_HOME is set
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("SUMO_HOME not set. Please set SUMO_HOME before running.")

sumocfg = os.path.abspath("data/sumocfgs/map1.sumocfg")

# Make sure log directory exists
log_path = os.path.abspath("data/sumocfgs/map1.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Start SUMO-GUI with TraCI
sumoCmd = ["sumo-gui", "-c", sumocfg, "--log", log_path]

traci.start(sumoCmd)

vehicles = []
stepsTotal = 3600

# Run simulation loop
step = 0
while step < stepsTotal:  # run for 3600 seconds
    traci.simulationStep()
    vehicles.append(traci.vehicle.getIDCount())
    step += 1

traci.close()

plt.plot(range(stepsTotal), vehicles)
plt.xlabel("Time (s)")
plt.ylabel("Vehicles in simulation")
plt.title("Vehicle count over time")
plt.show()