import os
import sys
import random
from keras.models import load_model
import traci
import matplotlib.pyplot as plt
import shutil
import numpy as np
import xml.etree.ElementTree as ET

# Carico Rete neurale
try:
    model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_simulation", "sumo_test", "final_generation_models", "individual_0.keras")    
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
    posV = t.vehicle.getPosition(vehicle_id)
    posP = t.person.getPosition(person_id)
    distance = t.simulation.getDistance2D(posV[0], posV[1], posP[0], posP[1], isDriving=True)

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


# --- Directory dei grafici ---
save_dir = "grafici"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, save_dir)

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
# --- Fine Directory dei grafici ---

# Inizializza SUMO
t = traci.connect(27910)

# Dati per i grafici
distance_data_t0 = {}
distance_data_t1 = {}
pair_names = {}
simulation_steps = []

lista_attivi = []
step_count = 0
simulation_time = 0
max_simulation_time = 40

# --- Genera lo scenario UNA SOLA VOLTA e ottieni il numero di pedoni ---
dbase = {}

for i in range(1, 10):
    num_pedestrians_scenario = defineScenario(dbase, "t_1")

# Modifica il file delle route dei pedoni
pedestrian_route_file = "TestCreazioneRete/trolleyNetPed.rou.xml"
pedestrian_route_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), pedestrian_route_file)
modify_pedestrian_routes(pedestrian_route_file, num_pedestrians_scenario)

scenario = np.array([
    dbase["t_1"]["nPassenger"],
    dbase["t_1"]["probPassenger"],
    dbase["t_1"]["nPedestrian"],
    dbase["t_1"]["probPedestrian"],
    dbase["t_1"]["altruism"]
])
scenario = scenario.reshape(1, 5)  # Reshape per il batch

print("Scenario:", scenario)

while simulation_time < max_simulation_time:

    l = t.simulationStep()
    step_count += 1
    simulation_time = t.simulation.getTime()
    collisions = t.simulation.getCollisions()

    for v1 in t.vehicle.getIDList():
        for p1 in t.person.getIDList():
            print("calcolo la distanza fra ", v1, " e ", p1)
            calculate_and_store_distance(v1, p1, step_count)

    for collision in collisions:
        collider_id = collision.__getattr__('collider')
        victim_id = collision.__getattr__('victim')

        if collider_id.startswith('vehicle') and victim_id.startswith('person'):
            posV = t.vehicle.getPosition(collider_id)
            posP = t.person.getPosition(victim_id)
            distance = t.simulation.getDistance2D(posV[0], posV[1], posP[0], posP[1], isDriving=True)
            print(f"Collisione! Veicolo: {collider_id}, Pedone: {victim_id}, Distanza: {distance}")
            print(f"Posizione Veicolo: {posV}, Posizione Pedone: {posP}")

    # Esempio di accesso ai dati di collisione (se necessario)
    if collisions:
        for collision in collisions:
            print(collision)

    # --- Controllo del veicolo t_1 con la rete neurale ---
    if model is not None and "t_1" in t.vehicle.getIDList():
        # Usa lo scenario generato all'inizio
        prediction = model.predict(scenario)

        # Interpreta la previsione e applica le azioni al veicolo
        # ***QUESTA PARTE DEVE ESSERE ADATTATA AL TUO MODELLO***
        # Esempio:
        acceleration = prediction[0][0]
        steering = prediction[0][1]

        target_speed = t.vehicle.getSpeed("t_1") + acceleration * 5
        t.vehicle.setSpeed("t_1", max(0, target_speed))

        # Esempio di sterzo (molto semplificato, vedi nota precedente)
        # t.vehicle.setAngle("t_1", steering * 10) # Non esiste setAngle. Serve changeLane o altro
    # --- Fine controllo veicolo t_1 ---

# Creazione grafici
if distance_data_t0:
    plot_distances(distance_data_t0, pair_names, simulation_steps, "Distanza Pedoni - Auto t_0", save_path)
if distance_data_t1:
    plot_distances(distance_data_t1, pair_names, simulation_steps, "Distanza Pedoni - Auto t_1", save_path)

t.close()