import os
import sys
import random
from keras.models import load_model
import traci
import sumolib
import matplotlib.pyplot as plt
import shutil
import numpy as np
import xml.etree.ElementTree as ET

# Load Neural Network
try:
    model_name_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_generation_models", "individual_40.keras")
    model = load_model(model_name_relative)
    print("Model successfully loaded from", model_name_relative)
    model.summary()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# Make sure SUMO_HOME is declared
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'share/sumo/tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def removeID(lista, x):
    """Removes element x from the list if present and not 0."""
    if (x != 0) and (x in lista):
        lista.remove(x)
    return lista

def plot_distances(distance_data, pair_names, simulation_steps, title, save_dir="grafici"):
    """
    Generates a plot of distances between vehicles and pedestrians and saves it in the specified directory,
    adding a marker when the distance is less than 10 meters.
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Time (simulation steps)")
    plt.ylabel("Distance (meters)")
    plt.title(title)
    plt.grid(True)

    if not distance_data:
        print(f"No distance data to plot for '{title}'.")
        return

    # Use simulation_steps_unique for the X axis
    if not simulation_steps:
        print(f"No simulation steps recorded for '{title}'.")
        return

    # Create an array of simulation steps for the X axis
    x_values = np.array(simulation_steps)

    # Define a sensible maximum distance for plotting to avoid extreme outliers
    MAX_PLOT_DISTANCE = 500.0 # meters

    for pair_key, data_points in distance_data.items():
        vehicle_name, person_name = pair_names[pair_key]
        distances = [dp[0] for dp in data_points]
        is_close_markers = [dp[1] for dp in data_points]

        distances_extended = np.full(len(x_values), np.nan)
        is_close_extended = np.full(len(x_values), False)

        for i in range(min(len(distances), len(x_values))):
            dist = distances[i]
            # Limit the distance for plotting if it's an extreme outlier or infinite
            if np.isinf(dist) or dist > MAX_PLOT_DISTANCE:
                distances_extended[i] = MAX_PLOT_DISTANCE # Limit to the maximum plot value
            else:
                distances_extended[i] = dist
            is_close_extended[i] = is_close_markers[i]

        plt.plot(x_values, distances_extended, label=f"Pedestrian: {person_name}")

        # Add markers for distances < 10m
        # Filter only the points where is_close_extended is True and the distance is not NaN
        close_indices = np.where(is_close_extended & ~np.isnan(distances_extended))[0]
        if len(close_indices) > 0:
            # To avoid duplicate labels in the legend, add the label only for the first occurrence
            # and use a temporary handle for the others.
            first_label_added = False
            for idx in close_indices:
                if not first_label_added:
                    plt.plot(x_values[idx], distances_extended[idx], 's', color='red', markersize=6)
                    first_label_added = True
                else:
                    plt.plot(x_values[idx], distances_extended[idx], 's', color='red', markersize=6)

    # To show only one label "Distance < 10m" in the legend, collect the handles
    # and unique labels and pass them to plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved in: {filepath}")

def plot_lane_occupancy(lane_data, simulation_steps, title, save_dir="grafici"):
    """
    Generates a plot of lane occupancy for vehicle t_0 at each simulation step
    and saves it in the specified directory.
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Time (simulation steps)")
    plt.ylabel("Lane number")
    plt.title(title)
    plt.grid(True)

    if not lane_data:
        print(f"No lane data to plot for '{title}'.")
        return

    if not simulation_steps:
        print(f"No simulation steps recorded for '{title}'.")
        return

    # Make sure lane data is aligned with simulation steps.
    # If there are fewer data points than simulation steps, it means the vehicle
    # may have left the simulation or was not present from the start.
    # In this case, extend with NaN to show interruptions.
    x_values = np.array(simulation_steps)
    lanes_extended = np.full(len(x_values), np.nan) # Initialize with NaN

    # Copy existing lane data into the corresponding steps
    for i in range(min(len(lane_data), len(x_values))):
        lanes_extended[i] = lane_data[i]

    # Plot the occupied lane. Use 'o' as a marker for each point.
    plt.plot(x_values, lanes_extended, marker='o', linestyle='-', markersize=4, label="Lane occupied by t_0")

    # Set Y axis ticks for integer lane numbers, if possible.
    # Avoid errors if there is no valid data.
    valid_lanes = [l for l in lanes_extended if not np.isnan(l)]
    if valid_lanes:
        min_lane = int(min(valid_lanes))
        max_lane = int(max(valid_lanes))
        plt.yticks(range(min_lane, max_lane + 1))
    else:
        # If there is no valid data, set a default range or let matplotlib decide.
        plt.yticks(range(0, 5)) # Example default range

    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved in: {filepath}")

def defineScenario(dbase, vehicleID):
    """Defines the parameters of a scenario for a given vehicle."""
    dbase[vehicleID] = {'nPassenger': random.randint(0, 5),
                        'probPassenger': random.uniform(0, 1),
                        'nPedestrian': random.randint(0, 5),
                        'probPedestrian': random.uniform(0, 1),
                        'altruism': random.uniform(0, 1),
                        }
    return dbase[vehicleID]

def calculate_and_store_distance(vehicle_id, person_id, step):
    """
    Calculates the distance between a vehicle and a pedestrian, stores it,
    and stores a flag if the distance is less than 10 meters.
    Handles extremely large distance values by setting them to 0.
    """
    global distance_data_t0, distance_data_t1, pair_names, simulation_steps_unique

    try:
        posV = traci.vehicle.getPosition(vehicle_id)
        posP = traci.person.getPosition(person_id)
        distance = traci.simulation.getDistance2D(posV[0], posV[1], posP[0], posP[1], isDriving=True)

        # Check if the distance is an extremely large value (similar to infinity)
        # or if it is greater than a reasonable threshold (e.g. 1000000000000.0)
        # If so, set it to 0 as requested.
        if distance > 1e15 or np.isinf(distance): # Using 1e15 as a very large number threshold
            distance = 0.0 # Set to 0 if the distance is extremely large or infinite

        is_close = distance < 10.0 # Flag for distance less than 10 meters

        pair_key = f"{vehicle_id}-{person_id}"
        if pair_key not in pair_names:
            pair_names[pair_key] = (vehicle_id, person_id)

        if vehicle_id == "t_0":
            if pair_key not in distance_data_t0:
                distance_data_t0[pair_key] = []
            distance_data_t0[pair_key].append((distance, is_close))
        elif vehicle_id == "t_1":
            if pair_key not in distance_data_t1:
                distance_data_t1[pair_key] = []
            distance_data_t1[pair_key].append((distance, is_close))

        # Add the current step to the list of unique steps (if not already present)
        if step not in simulation_steps_unique:
            simulation_steps_unique.append(step)
            simulation_steps_unique.sort() # Keeps steps sorted

    except traci.exceptions.TraCIException as e:
        pass # Ignore errors if vehicle/pedestrian is no longer present
    
def modify_pedestrian_routes(route_file, num_pedestrians):
    """Modifies the pedestrian route file according to the specified number."""
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()

        # Remove all existing pedestrians
        for person in root.findall('person'):
            root.remove(person)

        # Add new pedestrians
        for i in range(num_pedestrians):

            person_attrib = {
                "id": f"p_{i}",
                "depart": "0.00",
                "type": "fast_pedestrian",
                "jmIgnoreFoeProb": "1.0",  # flags to ignore rules on crossing
                "jmIgnoreFoeSpeed": "1.0",  
                "jmIgnoreJunctionFoeProb": "1.0",
                "jmCrossingGap": "0.0" 
            }
            person = ET.SubElement(root, "person", attrib=person_attrib)
            person_trip = ET.SubElement(person, "personTrip", attrib={"from": "-E4", "to": "E6"})


        ET.indent(tree, space="    ", level=0) # Format XML for readability
        tree.write(route_file, encoding="UTF-8", xml_declaration=True)
        print(f"Pedestrian route file '{route_file}' modified with {num_pedestrians} pedestrians.")

    except FileNotFoundError:
        print(f"Error: The file '{route_file}' was not found in the current directory.")
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing XML file '{route_file}': {e}")
        sys.exit(1)

import xml.etree.ElementTree as ET
import sys

def modify_vehicle_route_based_on_prediction(route_file_path, prediction_value):
    """
    Modifies the 'arrivalLane' attribute of vehicle 't_0' in the trip element
    of the route file based on the prediction value.

    Args:
        route_file_path (str): The path to the .rou.xml file.
        prediction_value (float): The prediction value (0 or 1, or between 0 and 1).
    """
    try:
        tree = ET.parse(route_file_path)
        root = tree.getroot()

        t0_trip = None
        for trip in root.findall('trip'):
            if trip.get('id') == 't_0':
                t0_trip = trip
                break
        
        if t0_trip is None:
            print(f"Error: Vehicle 't_0' trip not found in '{route_file_path}'.")
            return

        target_lane = None
        # Determine the target lane based on the prediction value
        if abs(prediction_value) <= 0.5: # Prediction is closer to 0
            target_lane = "2"
            print(f"DEBUG: Prediction is {prediction_value:.2f} (closer to 0). Setting t_0 arrivalLane to '{target_lane}'.")
        else: # Prediction is closer to 1
            target_lane = "1"
            print(f"DEBUG: Prediction is {prediction_value:.2f} (closer to 1). Setting t_0 arrivalLane to '{target_lane}'.")

        # Set the 'arrivalLane' attribute directly on the <trip> element
        t0_trip.set('arrivalLane', target_lane)
        
        # Format XML for readability and write changes back to the file
        ET.indent(tree, space="    ", level=0)
        tree.write(route_file_path, encoding="UTF-8", xml_declaration=True)
        print(f"Vehicle 't_0' arrivalLane successfully set to '{target_lane}' in '{route_file_path}'.")

    except FileNotFoundError:
        print(f"Error: The route file '{route_file_path}' was not found.")
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing XML file '{route_file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


# --- Directory for plots ---
save_dir = "grafici"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, save_dir)

# Remove and recreate the directory for clean plots at each run
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
# --- End Directory for plots ---


# Data for plots
distance_data_t0 = {}
distance_data_t1 = {}
pair_names = {}
simulation_steps_unique = [] # List to keep track of unique simulation steps
lane_data_t0 = [] # New list to store the lane index of t_0 at each step

step_count = 0
simulation_time = 0
max_simulation_time = 30

# --- Generate the scenario ONLY ONCE and get the number of pedestrians ---
dbase = {}
scenario_params_t1 = defineScenario(dbase, "t_1") # Renamed to avoid conflict with 'scenario' array
print(f"DEBUG: Number of pedestrians generated in the scenario: {scenario_params_t1['nPedestrian']}")

pedestrian_route_file = os.path.join(script_dir, "TestCreazioneRete", "trolleyNetPed.rou.xml")
modify_pedestrian_routes(pedestrian_route_file, scenario_params_t1['nPedestrian'])

scenario = np.array([ # This 'scenario' is the input for the NN
    dbase["t_1"]["nPassenger"],
    dbase["t_1"]["probPassenger"],
    dbase["t_1"]["nPedestrian"],
    dbase["t_1"]["probPedestrian"],
    dbase["t_1"]["altruism"]
])
scenario = scenario.reshape(1, 5)
print("Scenario for prediction:", scenario)
# --- End scenario generation ---

# Variables for calculating the probability of death for passengers/pedestrians
probDeath = None # Initialize to None, will be set by the prediction
prediction = None # Initialize to None, will be set by the prediction
t0_initial_setup_done = False # Flag to ensure speed mode/max speed are set only once for t_0

# --- Make the prediction BEFORE the simulation loop starts ---
if model is not None:
    prediction = model.predict(scenario, verbose=0)
    print(f"DEBUG: Initial prediction[0][0]: {prediction[0][0]}")
    
    # NEW: Call the function to modify the route file based on prediction
    car_route_file = os.path.join(script_dir, "TestCreazioneRete", "trolleyNetCar.rou.xml")
    modify_vehicle_route_based_on_prediction(car_route_file, prediction[0][0])

    if abs(prediction[0][0]) > 0.5: # choice = 1 => the car must turn
        probDeath = scenario[0][1] # set as death probability that of the passengers
        print("DEBUG: Prediction indicates car must turn (choice = 1).")
    else:
        probDeath = scenario[0][3] # set as death probability that of the pedestrians
        print("DEBUG: Prediction indicates car must go straight (choice = 0).")
else:
    print("Warning: Model is None, prediction will not be made. Defaulting probDeath to pedestrian scenario.")
    probDeath = scenario[0][3] # Default if no model is loaded
    # If no model, a default lane might be desired. Here, we'll set it to 1.
    car_route_file = os.path.join(script_dir, "TestCreazioneRete", "trolleyNetCar.rou.xml")
    modify_vehicle_route_based_on_prediction(car_route_file, 0.0) # Default to lane 2 if no prediction

# --- End of one-time prediction and route modification ---


# Initialize SUMO with traci.start (always in automatic mode)
sumo_cfg_file = os.path.join(script_dir, "TestCreazioneRete", "trolleyNet.sumocfg")

sumoCmd = [
    sumolib.checkBinary('sumo-gui'), # Usa sumo-gui per la modalità grafica
    "-c", sumo_cfg_file,
    # "--no-step-log", # Commenta o rimuovi per vedere i log nella GUI
    "--waiting-time-memory", "1000"
]

try:
    print("Starting SUMO simulation...")
    traci.start(sumoCmd)
    print("Connected to SUMO.")

    # --- Impostazioni per la visualizzazione GUI e la velocità della simulazione ---
    # Check if a GUI view is available before trying to configure it
    if traci.gui.getIDList():
        # zoom iniziale
        traci.gui.setZoom("View #0", 350) # "View #0" --> vista predefinita in SUMO-GUI
        traci.gui.setSchema("View #0", "real world") # O "pedestrians" per focalizzarsi sui pedoni
    else:
        print("DEBUG: No GUI view available. Skipping GUI settings.")


    # Imposta la velocità di simulazione. Un valore di 1.0 significa 1 secondo di simulazione = 1 secondo reale.
    traci.simulation.setScale(0.01) # Rallenta la simulazione di 10 volte per una migliore visualizzazione
    # --- Fine impostazioni GUI ---

    # Main simulation loop
    while traci.simulation.getMinExpectedNumber() > 0 and simulation_time < max_simulation_time:

        traci.simulationStep()
        step_count += 1
        simulation_time = traci.simulation.getTime()

        # Add the current step to the list of unique steps (if not already present)
        if step_count not in simulation_steps_unique:
            simulation_steps_unique.append(step_count)
            simulation_steps_unique.sort() # Keeps the list of steps sorted

        # --- Handle t_0 vehicle setup (speed mode, max speed) ---
        if "t_0" in traci.vehicle.getIDList():
            vehID_t0 = "t_0"
            if not t0_initial_setup_done:
                try:
                    # Imposta la velocità massima (m/s). Un valore molto alto per non limitare.
                    traci.vehicle.setMaxSpeed(vehID_t0, 50.0)
                    # Disabilita i controlli di velocità interni di SUMO per t_0
                    traci.vehicle.setSpeedMode(vehID_t0, 0) # 0 = all checks disabled (TraCI controls speed)
                    # Forzare i cambi di corsia ignorando i controlli di sicurezza e cooperazione
                    traci.vehicle.setSpeed(vehID_t0, 10)
                    traci.vehicle.setLaneChangeMode(vehID_t0, 1) # 1 = no safety checks, no cooperativeness
                    t0_initial_setup_done = True
                    print(f"DEBUG: {vehID_t0} speed mode and lane change mode set.")
                except traci.exceptions.TraCIException as e:
                    print(f"DEBUG: ERRORE TraCI durante l'inizializzazione di {vehID_t0}: {e}")
                    pass # Riproverà nello step successivo (la flag non è stata ancora settata a True)

            # --- Record lane occupancy for vehicle t_0 ---
            try:
                lane_index = traci.vehicle.getLaneIndex(vehID_t0)
                lane_data_t0.append(lane_index)
            except traci.exceptions.TraCIException:
                lane_data_t0.append(np.nan)
        else:
            # Se t_0 non è ancora in simulazione o è uscito
            lane_data_t0.append(np.nan)
        # --- End t_0 vehicle setup ---


        current_vehicles = traci.vehicle.getIDList()
        current_persons = traci.person.getIDList()

        for v1 in traci.vehicle.getIDList():
            for p1 in traci.person.getIDList():
                calculate_and_store_distance(v1, p1, step_count)
        
        # --- Pedestrian behavior modification logic (commented out as per previous instructions to force collisions) ---
        # NESSUN CODICE DI MODIFICA VELOCITÀ PEDONI QUI

    scenarioDice = random.random()
    print("Dice = ", scenarioDice, "probDeath = ", probDeath, "Prediction: ", round(abs(prediction[0][0])))

    # Here you enter if the prediction is = 1
    if (probDeath == scenario[0][1]):
        if(scenarioDice < probDeath):
            print(" PASSENGERS DEAD ")
        else:
            print(" PASSENGERS SAFE ")

    # Here you enter if the prediction is = 0
    if (probDeath == scenario[0][3]):
        if(scenarioDice < probDeath):
            print(" PEDESTRIANS DEAD ")
        else:
            print(" PEDESTRIANS SAFE ")

    print(f"Simulation ended after {step_count} steps or {simulation_time:.2f} seconds.")

except traci.exceptions.TraCIException as e:
    print(f"TraCI error: {e}")
    print("Make sure SUMO is started correctly and that the .sumocfg file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    try:
        if traci.is_connected():
            traci.close()
            print("SUMO connection closed.")
        else:
            print("No active SUMO connection to close (already closed or not connected).")
    except AttributeError:
        try:
            traci.close()
            print("SUMO connection closed (handling for older TraCI version).")
        except traci.exceptions.FatalTraCIError:
            print("No active SUMO connection to close.")
        except Exception as e:
            print(f"Error while closing TraCI connection: {e}")

# Create plots AFTER closing TraCI
if distance_data_t0:
    plot_distances(distance_data_t0, pair_names, simulation_steps_unique, "Distance Pedestrians - Car t_0 (AV)", save_path)
else:
    print("DEBUG: distance_data_t0 is empty. The distance plot for t_0 will not be generated.")

if distance_data_t1:
    plot_distances(distance_data_t1, pair_names, simulation_steps_unique, "Distance Pedestrians - Car t_1 (OBSTACLE)", save_path)

if lane_data_t0:
    plot_lane_occupancy(lane_data_t0, simulation_steps_unique, "Lane Occupancy Car t_0 (AV)", save_path)
else:
    print("DEBUG: lane_data_t0 is empty. The lane occupancy plot for t_0 will not be generated.")

print("Script completed.")