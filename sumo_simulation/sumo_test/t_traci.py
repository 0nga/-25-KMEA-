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
    model_name_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_generation_models", "individual_0.keras")
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
    return dbase[vehicleID]['nPedestrian']


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
            # DEBUG: Print when a distance data point for t_0 is added
            # print(f"DEBUG: Distance for t_0 and {person_id} at step {step}: {distance:.2f}m (Close: {is_close})")
        elif vehicle_id == "t_1":
            if pair_key not in distance_data_t1:
                distance_data_t1[pair_key] = []
            distance_data_t1[pair_key].append((distance, is_close))

        # Add the current step to the list of unique steps (if not already present)
        if step not in simulation_steps_unique:
            simulation_steps_unique.append(step)
            simulation_steps_unique.sort() # Keeps steps sorted

    except traci.exceptions.TraCIException as e:
        # print(f"TraCI error during distance calculation: {e}") # Debugging
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
            person = ET.SubElement(root, "person", id=f"p_{i}", depart="0.00", type="fast_pedestrian")
            person_trip = ET.SubElement(person, "personTrip", attrib={"from": "-E4", "to": "E6"}) # Changed 'from' and 'to' based on trolleyNet.net.xml

        ET.indent(tree, space="    ", level=0) # Format XML for readability
        tree.write(route_file, encoding="UTF-8", xml_declaration=True)
        print(f"Pedestrian route file '{route_file}' modified with {num_pedestrians} pedestrians.")

    except FileNotFoundError:
        print(f"Error: The file '{route_file}' was not found in the current directory.")
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing XML file '{route_file}': {e}")
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

lista_attivi = [] # This variable does not seem to be used, can be removed if not needed
step_count = 0
simulation_time = 0
max_simulation_time = 300

# --- Generate the scenario ONLY ONCE and get the number of pedestrians ---
dbase = {}
num_pedestrians_scenario = defineScenario(dbase, "t_1")
# DEBUG: Print the number of pedestrians generated
print(f"DEBUG: Number of pedestrians generated in the scenario: {num_pedestrians_scenario}")

pedestrian_route_file = "TestCreazioneRete/trolleyNetPed.rou.xml"
pedestrian_route_file_abs = os.path.join(script_dir, pedestrian_route_file)
modify_pedestrian_routes(pedestrian_route_file_abs, num_pedestrians_scenario)

scenario = np.array([
    dbase["t_1"]["nPassenger"],
    dbase["t_1"]["probPassenger"],
    dbase["t_1"]["nPedestrian"],
    dbase["t_1"]["probPedestrian"],
    dbase["t_1"]["altruism"]
])
scenario = scenario.reshape(1, 5)
print("Scenario:", scenario)
# --- End scenario generation ---

# Variables for calculating the probability of death for passengers/pedestrians
probDeath = scenario[0][3] # by default I leave that of pedestrians (car straight - choice = 0)

# Initialize SUMO with traci.start (always in automatic mode)
sumo_cfg_file = os.path.join(script_dir, "TestCreazioneRete", "trolleyNet.sumocfg")

sumoCmd = [
    sumolib.checkBinary('sumo-gui'), # CAMBIATO: Usa sumo-gui per la modalità grafica
    "-c", sumo_cfg_file,
    # "--no-step-log", # Commenta o rimuovi per vedere i log nella GUI
    "--waiting-time-memory", "1000"
]

try:
    print("Starting SUMO simulation...")
    traci.start(sumoCmd)
    print("Connected to SUMO.")

    # --- Impostazioni per la visualizzazione GUI e la velocità della simulazione ---
    # Imposta lo zoom iniziale per avere una buona panoramica (regola i valori)
    traci.gui.setZoom("View #0", 1000) # "View #0" è il nome della vista predefinita in SUMO-GUI
    # Puoi anche impostare uno schema di visualizzazione se vuoi, es. per mostrare le persone
    traci.gui.setSchema("View #0", "real world") # O "pedestrians" per focalizzarsi sui pedoni

    # Imposta la velocità di simulazione. Un valore di 1.0 significa 1 secondo di simulazione = 1 secondo reale.
    # Valori inferiori a 1.0 rallentano la simulazione (es. 0.1 è 10x più lento).
    # Valori superiori a 1.0 la velocizzano.
    traci.simulation.setScale(0.1) # CAMBIATO: Rallenta la simulazione di 10 volte per una migliore visualizzazione
    # --- Fine impostazioni GUI ---


    # --- Pedestrian behavior modification variables ---
    default_ped_speed = 1.3 # meters/second (adjust as needed)
    collision_threshold_distance = 2.0 # meters, adjust based on desired behavior
    # --- End pedestrian behavior modification variables ---

    # Main simulation loop
    while traci.simulation.getMinExpectedNumber() > 0 and simulation_time < max_simulation_time:

        traci.simulationStep()
        step_count += 1
        simulation_time = traci.simulation.getTime()

        # Add the current step to the list of unique steps (if not already present)
        if step_count not in simulation_steps_unique:
            simulation_steps_unique.append(step_count)
            simulation_steps_unique.sort() # Keeps the list of steps sorted

        # --- Record lane occupancy for vehicle t_0 ---
        if "t_0" in traci.vehicle.getIDList():
            try:
                traci.vehicle.setMaxSpeed("t_0", 3.8) # Imposta la velocità massima a 0.5 m/s
                lane_index = traci.vehicle.getLaneIndex("t_0")
                lane_data_t0.append(lane_index)
            except traci.exceptions.TraCIException:
                # If t_0 is present in the ID list but does not have a valid position (e.g. just started/arrived)
                lane_data_t0.append(np.nan) # Add NaN to indicate missing data
        else:
            # If t_0 is not present in the simulation at this step
            lane_data_t0.append(np.nan) # Add NaN to keep the length aligned with steps
        # --- End lane occupancy recording ---

        # DEBUG: Print active vehicles and pedestrians at each step
        current_vehicles = traci.vehicle.getIDList()
        current_persons = traci.person.getIDList()
        # print(f"DEBUG: Step {step_count}: Active vehicles: {current_vehicles}, Active pedestrians: {current_persons}")


        for vehID in traci.simulation.getDepartedIDList():
            if traci.vehicle.getTypeID(vehID) == "reckless":
                traci.vehicle.setSpeedMode(vehID, 0)
                traci.vehicle.setSpeed(vehID, 15)

        for v1 in traci.vehicle.getIDList():
            for p1 in traci.person.getIDList():
                calculate_and_store_distance(v1, p1, step_count)
        
        # --- Start Pedestrian behavior modification logic ---
        ped_ids = traci.person.getIDList()
        car_ids = traci.vehicle.getIDList()

        for ped_id in ped_ids:
            current_ped_speed = traci.person.getSpeed(ped_id)
            ped_x, ped_y = traci.person.getPosition(ped_id)

            should_force_move = True # Assume pedestrian should try to keep moving

            for car_id in car_ids:
                try:
                    car_x, car_y = traci.vehicle.getPosition(car_id)
                    distance = ((ped_x - car_x)**2 + (ped_y - car_y)**2)**0.5

                    if distance < collision_threshold_distance:
                        # A car is very close, try to force the pedestrian to continue.
                        # This might lead to collisions if the threshold is too small or if
                        # the pedestrian's path directly intersects the car's path without avoidance.
                        # print(f"DEBUG: Pedestrian {ped_id} close to car {car_id} ({distance:.2f}m). Forcing speed.")
                        traci.person.setSpeed(ped_id, default_ped_speed)
                        should_force_move = False # Handled this pedestrian due to car proximity
                        break # No need to check other cars for this pedestrian
                except traci.exceptions.TraCIException:
                    # Car might have left simulation or is not yet active, ignore error
                    pass

            if should_force_move and current_ped_speed < default_ped_speed:
                # If no car is too close and the pedestrian is moving slower than desired,
                # set their speed to the default walking speed.
                traci.person.setSpeed(ped_id, default_ped_speed)
        # --- End Pedestrian behavior modification logic ---

        if model is not None and "t_0" in traci.vehicle.getIDList():
            prediction = model.predict(scenario, verbose=0)

            if abs(prediction[0][0]) > 0.5: # choice = 1 => the car must turn

                # set as death probability that of the passengers
                probDeath = scenario[0][1]                     

                try:
                    current_lane_index = traci.vehicle.getLaneIndex("t_0")
                    current_edge_id = traci.vehicle.getRoadID("t_0")

                    if not current_edge_id.startswith(":"):
                        num_lanes_on_edge = traci.edge.getLaneNumber(current_edge_id)

                        if current_lane_index + 1 < num_lanes_on_edge:
                            traci.vehicle.changeLane("t_0", current_lane_index + 1, 0.5)
                except traci.exceptions.TraCIException as e:
                    pass # Ignore errors if the lane is not valid or the vehicle is no longer there
            else:
                # set as death probability that of the pedestrians
                probDeath = scenario[0][3]

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