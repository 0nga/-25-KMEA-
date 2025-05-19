import os
import sys
import random
from keras.models import load_model
import traci
import sumolib  # Import sumolib
import matplotlib.pyplot as plt
import shutil
import numpy as np
import xml.etree.ElementTree as ET

# Carico Rete neurale
try:
    model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_generation_models", "individual_0.keras")
    model = load_model(model_name)
    print("Modello caricato con successo da", model_name)
    model.summary()

except Exception as e:
    print(f"Si è verificato un errore durante il caricamento del modello: {e}")
    model = None

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'share/sumo/tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def removeID(lista, x):
    if (x != 0) and (x in lista):
        lista.remove(x)
    return lista


def plot_distances(distance_data, pair_names, simulation_steps, title, save_dir="grafici"):
    """Genera un grafico e lo salva nella directory specificata."""
    plt.figure(figsize=(10, 6))
    plt.xlabel("Tempo (passi di simulazione)")
    plt.ylabel("Distanza (metri)")
    plt.title(title)
    plt.grid(True)

    if not distance_data:
        print("Nessun dato di distanza da graficare.")
        return

    max_steps = len(simulation_steps)
    x_values = range(1, max_steps + 1)

    for pair_key, distances in distance_data.items():
        vehicle_name, person_name = pair_names[pair_key]
        if len(distances) < max_steps:
            last_distance = distances[-1]
            distances_extended = distances + [last_distance] * (max_steps - len(distances))
        else:
            distances_extended = distances
        plt.plot(x_values, distances_extended, label=f"Pedone: {person_name}")

    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Grafico salvato in: {filepath}")



def defineScenario(dbase, vehicleID):
    dbase[vehicleID] = {'nPassenger': random.randint(0, 5),
                        'probPassenger': random.uniform(0, 1),
                        'nPedestrian': random.randint(0, 5),
                        'probPedestrian': random.uniform(0, 1),
                        'altruism': random.uniform(0, 1),
                        }
    return dbase[vehicleID]['nPedestrian'] # Restituisci il numero di pedoni



def calculate_and_store_distance(vehicle_id, person_id, step):
    """Calcola la distanza tra un veicolo e un pedone e la memorizza."""
    global distance_data_t0, distance_data_t1, pair_names, simulation_steps
    posV = traci.vehicle.getPosition(vehicle_id)
    posP = traci.person.getPosition(person_id)
    distance = traci.simulation.getDistance2D(posV[0], posV[1], posP[0], posP[1], isDriving=True)

    pair_key = f"{vehicle_id}-{person_id}"
    if pair_key not in pair_names:
        pair_names[pair_key] = (vehicle_id, person_id)

    if vehicle_id == "t_0":
        if pair_key not in distance_data_t0:
            distance_data_t0[pair_key] = []
        distance_data_t0[pair_key].append(distance)
    elif vehicle_id == "t_1":
        if pair_key not in distance_data_t1:
            distance_data_t1[pair_key] = []
        distance_data_t1[pair_key].append(distance)

    simulation_steps.append(step)



def modify_pedestrian_routes(route_file, num_pedestrians):
    """Modifica il file di route dei pedoni in base al numero specificato."""
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()

        # Rimuovi tutti gli elementi 'person' esistenti
        for person in root.findall('person'):
            root.remove(person)

        # Aggiungi nuovi elementi 'person' in base a num_pedestrians
        for i in range(num_pedestrians):
            person = ET.SubElement(root, "person", id=f"p_{i}", depart="0.00")
            person_trip = ET.SubElement(person, "personTrip", attrib={"from": "-E4", "to": "E6"})

        # Scrivi le modifiche nel file (sovrascrivendo l'originale)
        tree.write(route_file, encoding="UTF-8", xml_declaration=True)
        print(f"File delle route dei pedoni '{route_file}' modificato con {num_pedestrians} pedoni.")

    except FileNotFoundError:
        print(f"Errore: Il file '{route_file}' non è stato trovato nella directory corrente.")
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Errore durante l'analisi del file XML '{route_file}': {e}")
        sys.exit(1)



# --- Directory dei grafici ---
save_dir = "grafici"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, save_dir)

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
# --- Fine Directory dei grafici ---

# Inizializza SUMO
# t = traci.connect(27910) # Removed: Start SUMO with traci.start

traci.start([sumolib.checkBinary('sumo'),
            "-n", "TestCreazioneRete/trolleyNet.net.xml",  # Replace with your network file
            "-r", "TestCreazioneRete/trolleyNetCar.rou.xml, TestCreazioneRete/trolleyNetPed.rou.xml ",  # Replace with your routes file
            "--collision.check-junctions",
            "--collision-output", "collisions.xml",
            "--no-step-log"
            ])
# --- End of incorporated SUMO startup ---

# Dati per i grafici
distance_data_t0 = {}
distance_data_t1 = {}
pair_names = {}
simulation_steps = []

lista_attivi = []
step_count = 0
simulation_time = 0
max_simulation_time = 40  # Set a maximum simulation time

# --- Genera lo scenario UNA SOLA VOLTA e ottieni il numero di pedoni ---
dbase = {}
num_pedestrians_scenario = defineScenario(dbase, "t_1")

# Modifica il file delle route dei pedoni
pedestrian_route_file = "TestCreazioneRete/trolleyNetPed.rou.xml"  # Adjust the path if needed
pedestrian_route_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), pedestrian_route_file) #make path absolute
modify_pedestrian_routes(pedestrian_route_file, num_pedestrians_scenario)

# --- Carica il modello con percorso relativo ---
try:
    model_name_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_generation_models", "individual_0.keras")
    model = load_model(model_name_relative)
    print("Modello caricato con successo da", model_name_relative)
    model.summary()
except Exception as e:
    print(f"Si è verificato un errore durante il caricamento del modello: {e}")
    model = None

scenario = np.array([
    dbase["t_1"]["nPassenger"],
    dbase["t_1"]["probPassenger"],
    dbase["t_1"]["nPedestrian"],
    dbase["t_1"]["probPedestrian"],
    dbase["t_1"]["altruism"]
])
scenario = scenario.reshape(1, 5)  # Reshape per il batch

print("Scenario:", scenario)
# --- Fine generazione scenario ---

# traci.simulationStep() # Removed, already in traci.start

while traci.simulation.getMinExpectedNumber() > 0 and simulation_time < max_simulation_time:
    traci.simulationStep()
    step_count += 1
    simulation_time = traci.simulation.getTime()
    collisions = traci.simulation.getCollisions()

    # --- Vehicle control from runner.py ---
    for vehID in traci.simulation.getDepartedIDList():
        if traci.vehicle.getTypeID(vehID) == "reckless":
            traci.vehicle.setSpeedMode(vehID, 0)
            traci.vehicle.setSpeed(vehID, 15)
    # --- End of vehicle control ---

    for v1 in traci.vehicle.getIDList():
        for p1 in traci.person.getIDList():
            # print("calcolo la distanza fra ", v1, " e ", p1)
            calculate_and_store_distance(v1, p1, step_count)

    for collision in collisions:
        collider_id = collision.collider
        victim_id = collision.victim

        # Improved collision detection
        if (traci.vehicle.isVehicle(collider_id) and traci.person.isPerson(victim_id)) or \
           (traci.person.isPerson(collider_id) and traci.vehicle.isVehicle(victim_id)):
            posV = traci.vehicle.getPosition(collider_id)
            posP = traci.person.getPosition(victim_id)
            distance = traci.simulation.getDistance2D(posV[0], posV[1], posP[0], posP[1], isDriving=True)
            print(f"Collisione! Veicolo: {collider_id}, Pedone: {victim_id}, Distanza: {distance}")
            print(f"Posizione Veicolo: {posV}, Posizione Pedone: {posP}")

    # Esempio di accesso ai dati di collisione (se necessario)
    if collisions:
        for collision in collisions:
            print(collision)

    # --- Controllo del veicolo t_1 con la rete neurale ---
    if model is not None and "t_1" in traci.vehicle.getIDList():
        # Usa lo scenario generato all'inizio
        prediction = model.predict(scenario)

        # Interpreta la previsione e applica le azioni al veicolo: DA MODIFICARE
        # print(f"Prediction: {prediction}") #useful for debug
        if abs(prediction[0][0]) < 0.5:  # Example threshold, adjust as needed
            try:
                traci.vehicle.changeLane("t_1", 1, 0.5)  # Use a safe lane change
            except:
                print("lane change failed")
        elif abs(prediction[0][0]) >= 0.5:
            pass

        print("Prediction: ", prediction)
    # --- Fine controllo veicolo t_1 ---

# Creazione grafici
if distance_data_t0:
    plot_distances(distance_data_t0, pair_names, simulation_steps, "Distanza Pedoni - Auto t_0", save_path)
if distance_data_t1:
    plot_distances(distance_data_t1, pair_names, simulation_steps, "Distanza Pedoni - Auto t_1", save_path)

traci.close()

