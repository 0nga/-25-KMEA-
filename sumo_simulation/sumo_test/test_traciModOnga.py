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

# Carico Rete neurale
try:
    model_name_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_generation_models", "individual_0.keras")
    model = load_model(model_name_relative)
    print("Modello caricato con successo da", model_name_relative)
    model.summary()
except Exception as e:
    print(f"Si è verificato un errore durante il caricamento del modello: {e}")
    model = None

# Assicurati che SUMO_HOME sia dichiarato
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
    """Genera un grafico e lo salva nella directory specificata,
    aggiungendo un marcatore quando la distanza è inferiore a 10 metri.""" # Modificato il testo della docstring
    plt.figure(figsize=(10, 6))
    plt.xlabel("Tempo (passi di simulazione)")
    plt.ylabel("Distanza (metri)")
    plt.title(title)
    plt.grid(True)

    if not distance_data:
        print(f"Nessun dato di distanza da graficare per '{title}'.")
        return

    if not simulation_steps:
        print(f"Nessun passo di simulazione registrato per '{title}'.")
        return

    max_steps = len(simulation_steps)
    x_values = range(1, max_steps + 1)

    for pair_key, data_points in distance_data.items(): # data_points ora è una lista di tuple (distance, is_close)
        vehicle_name, person_name = pair_names[pair_key]
        distances = [dp[0] for dp in data_points]
        is_close_markers = [dp[1] for dp in data_points]

        if len(distances) < max_steps:
            last_distance = distances[-1] if distances else 0
            distances_extended = distances + [last_distance] * (max_steps - len(distances))
            is_close_extended = is_close_markers + [False] * (max_steps - len(is_close_markers)) # Pad with False
        else:
            distances_extended = distances[:max_steps]
            is_close_extended = is_close_markers[:max_steps]

        plt.plot(x_values, distances_extended, label=f"Pedone: {person_name}")

        # Aggiungi i marcatori per le distanze < 10m
        for i, (dist, is_close) in enumerate(zip(distances_extended, is_close_extended)):
            if is_close:
                # Usiamo 's' per quadrato, e mettiamo la label solo per la prima occorrenza per evitare duplicati nella legenda
                plt.plot(x_values[i], dist, 's', color='red', markersize=6, label=f"Distanza < 10m per {person_name}" if i == 0 else "")

    # Per mostrare una sola etichetta "Distanza < 10m" nella legenda, raccogliamo le maniglie
    # e le etichette uniche e le passiamo a plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

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
    return dbase[vehicleID]['nPedestrian']


def calculate_and_store_distance(vehicle_id, person_id, step):
    """Calcola la distanza tra un veicolo e un pedone e la memorizza.
    Inoltre, memorizza un flag se la distanza è inferiore a 10 metri.""" # Modificato il testo della docstring
    global distance_data_t0, distance_data_t1, pair_names, simulation_steps_unique

    try:
        posV = traci.vehicle.getPosition(vehicle_id)
        posP = traci.person.getPosition(person_id)
        distance = traci.simulation.getDistance2D(posV[0], posV[1], posP[0], posP[1], isDriving=True)
        is_close = distance < 10.0 # Flag per distanza inferiore a 10 metri

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

        if step not in simulation_steps_unique:
            simulation_steps_unique.append(step)

    except traci.exceptions.TraCIException as e:
        pass


def modify_pedestrian_routes(route_file, num_pedestrians):
    """Modifica il file di route dei pedoni in base al numero specificato."""
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()

        for person in root.findall('person'):
            root.remove(person)

        for i in range(num_pedestrians):
            person = ET.SubElement(root, "person", id=f"p_{i}", depart="0.00", type="fast_pedestrian")
            person_trip = ET.SubElement(person, "personTrip", attrib={"from": "-E4", "to": "E6"})

        ET.indent(tree, space="    ", level=0)
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


# Dati per i grafici
distance_data_t0 = {}
distance_data_t1 = {}
pair_names = {}
simulation_steps_unique = []

lista_attivi = []
step_count = 0
simulation_time = 0
max_simulation_time = 3600

# --- Genera lo scenario UNA SOLA VOLTA e ottieni il numero di pedoni ---
dbase = {}
num_pedestrians_scenario = defineScenario(dbase, "t_1")

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
# --- Fine generazione scenario ---


# Chiedi all'utente la modalità di simulazione
print("\nSeleziona la modalità di simulazione:")
print("1. Simulazione automatica (termina da sola)")
print("2. Simulazione passo-passo (premi un tasto per avanzare)")
mode = input("Inserisci il numero della modalità (1 o 2): ")

while mode not in ['1', '2']:
    print("Input non valido. Inserisci 1 o 2.")
    mode = input("Inserisci il numero della modalità (1 o 2): ")

# Inizializza SUMO con traci.start
sumo_cfg_file = os.path.join(script_dir, "TestCreazioneRete", "trolleyNet.sumocfg")

sumoCmd = [
    sumolib.checkBinary('sumo-gui') if mode == '2' else sumolib.checkBinary('sumo'), # Usa sumo-gui per la modalità passo-passo
    "-c", sumo_cfg_file,
    "--no-step-log",
    "--waiting-time-memory", "1000"
]

try:
    print("Avvio simulazione SUMO...")
    traci.start(sumoCmd)
    print("Connesso a SUMO.")

    # Loop principale della simulazione
    while traci.simulation.getMinExpectedNumber() > 0 and simulation_time < max_simulation_time:
        if mode == '2':
            input(f"Premi INVIO per il passo {step_count + 1} (o chiudi la finestra SUMO per terminare)...")
            if not traci.is_connected(): # L'utente potrebbe aver chiuso SUMO-GUI
                print("SUMO-GUI chiuso dall'utente. Terminazione simulazione.")
                break

        traci.simulationStep()
        step_count += 1
        simulation_time = traci.simulation.getTime()

        for vehID in traci.simulation.getDepartedIDList():
            if traci.vehicle.getTypeID(vehID) == "reckless":
                traci.vehicle.setSpeedMode(vehID, 0)
                traci.vehicle.setSpeed(vehID, 15)

        for v1 in traci.vehicle.getIDList():
            for p1 in traci.person.getIDList():
                calculate_and_store_distance(v1, p1, step_count)

        if model is not None and "t_0" in traci.vehicle.getIDList():
            prediction = model.predict(scenario, verbose=0)

            if prediction[0][0] > 0.5:
                try:
                    current_lane_index = traci.vehicle.getLaneIndex("t_0")
                    current_edge_id = traci.vehicle.getRoadID("t_0")

                    if not current_edge_id.startswith(":"):
                        num_lanes_on_edge = traci.edge.getLaneNumber(current_edge_id)

                        if current_lane_index + 1 < num_lanes_on_edge:
                            traci.vehicle.changeLane("t_0", current_lane_index + 1, 0.5)
                except traci.exceptions.TraCIException as e:
                    pass

    print(f"Simulazione terminata dopo {step_count} passi o {simulation_time:.2f} secondi.")

except traci.exceptions.TraCIException as e:
    print(f"Errore TraCI: {e}")
    print("Assicurati che SUMO sia avviato correttamente e che il percorso del file .sumocfg sia corretto.")
except Exception as e:
    print(f"Si è verificato un errore inaspettato: {e}")
finally:
    try:
        if traci.is_connected():
            traci.close()
            print("Connessione SUMO chiusa.")
        else:
            print("Nessuna connessione SUMO attiva da chiudere (già chiusa o non connessa).")
    except AttributeError:
        try:
            traci.close()
            print("Connessione SUMO chiusa (gestione per versione TraCI meno recente).")
        except traci.exceptions.FatalTraCIError:
            print("Nessuna connessione SUMO attiva da chiudere.")
        except Exception as e:
            print(f"Errore durante la chiusura della connessione TraCI: {e}")

# Creazione grafici DOPO la chiusura di TraCI
if distance_data_t0:
    plot_distances(distance_data_t0, pair_names, simulation_steps_unique, "Distanza Pedoni - Auto t_0 (AV)", save_path)
if distance_data_t1:
    plot_distances(distance_data_t1, pair_names, simulation_steps_unique, "Distanza Pedoni - Auto t_1 (OSTACOLO)", save_path)

print("Script completato.")