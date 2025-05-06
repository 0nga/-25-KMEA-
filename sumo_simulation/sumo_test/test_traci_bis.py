from __future__ import print_function
from __future__ import absolute_import
import os
import sys

SUMO_HOME = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
sys.path.append(os.path.join(os.environ.get("SUMO_HOME", SUMO_HOME), "tools"))
import traci  # noqa
import sumolib  # noqa

traci.start([sumolib.checkBinary('sumo'),
    "-n", "input_net3.net.xml",
    "-r", "input_routes.rou.xml",
    "--no-step-log"
    ])
traci.simulationStep()

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    for vehID in traci.simulation.getDepartedIDList():
        if traci.vehicle.getTypeID(vehID) == "reckless":
            traci.vehicle.setSpeedMode(vehID, 0)
            traci.vehicle.setSpeed(vehID, 15)

    collisions = traci.simulation.getCollisions()
    if len(collisions) > 0:
        print(traci.simulation.getTime(), collisions)

traci.close()