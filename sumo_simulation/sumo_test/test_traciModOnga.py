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
    """Rimuove un elemento x dalla lista se presente e non è 0."""
    if (x != 0) and (x in lista):
        lista.remove(x)
    return lista


def plot_distances(distance_data, pair_names, simulation_steps, title, save_dir="grafici"):
    """
    Genera un grafico delle distanze tra veicoli e pedoni e lo salva nella directory specificata,
    aggiungendo un marcatore quando la distanza è inferiore a 10 metri.
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Tempo (passi di simulazione)")
    plt.ylabel("Distanza (metri)")
    plt.title(title)
    plt.grid(True)

    if not distance_data:
        print(f"Nessun dato di distanza da graficare per '{title}'.")
        return

    # Utilizziamo simulation_steps_unique per l'asse X
    if not simulation_steps:
        print(f"Nessun passo di simulazione registrato per '{title}'.")
        return

    # Creiamo un array di passi di simulazione per l'asse X
    x_values = np.array(simulation_steps)

    # Definisci una distanza massima sensata per il plotting per evitare outlier estremi
    MAX_PLOT_DISTANCE = 500.0 # metri

    for pair_key, data_points in distance_data.items():
        vehicle_name, person_name = pair_names[pair_key]
        distances = [dp[0] for dp in data_points]
        is_close_markers = [dp[1] for dp in data_points]

        distances_extended = np.full(len(x_values), np.nan)
        is_close_extended = np.full(len(x_values), False)

        for i in range(min(len(distances), len(x_values))):
            dist = distances[i]
            # Limita la distanza per il plotting se è un outlier estremo o infinito
            if np.isinf(dist) or dist > MAX_PLOT_DISTANCE:
                distances_extended[i] = MAX_PLOT_DISTANCE # Limita al valore massimo di plot
            else:
                distances_extended[i] = dist
            is_close_extended[i] = is_close_markers[i]


        plt.plot(x_values, distances_extended, label=f"Pedone: {person_name}")

        # Aggiungi i marcatori per le distanze < 10m
        # Filtra solo i punti in cui is_close_extended è True e la distanza non è NaN
        close_indices = np.where(is_close_extended & ~np.isnan(distances_extended))[0]
        if len(close_indices) > 0:
            # Per evitare etichette duplicate nella legenda, aggiungiamo la label solo per la prima occorrenza
            # e usiamo un handle temporaneo per le successive.
            first_label_added = False
            for idx in close_indices:
                if not first_label_added:
                    plt.plot(x_values[idx], distances_extended[idx], 's', color='red', markersize=6, label=f"Distanza < 10m per {person_name}")
                    first_label_added = True
                else:
                    plt.plot(x_values[idx], distances_extended[idx], 's', color='red', markersize=6)


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


def plot_lane_occupancy(lane_data, simulation_steps, title, save_dir="grafici"):
    """
    Genera un grafico dell'occupazione della corsia per il veicolo t_0 ad ogni passo
    della simulazione e lo salva nella directory specificata.
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Tempo (passi di simulazione)")
    plt.ylabel("Numero di corsia")
    plt.title(title)
    plt.grid(True)

    if not lane_data:
        print(f"Nessun dato di corsia da graficare per '{title}'.")
        return

    if not simulation_steps:
        print(f"Nessun passo di simulazione registrato per '{title}'.")
        return

    # Assicurati che i dati della corsia siano allineati con i passi di simulazione.
    # Se ci sono meno punti dati che passi di simulazione, significa che il veicolo
    # potrebbe essere uscito dalla simulazione o non essere stato presente dall'inizio.
    # In questo caso, estendiamo con NaN per mostrare le interruzioni.
    x_values = np.array(simulation_steps)
    lanes_extended = np.full(len(x_values), np.nan) # Inizializza con NaN

    # Copia i dati di corsia esistenti nei passi corrispondenti
    for i in range(min(len(lane_data), len(x_values))):
        lanes_extended[i] = lane_data[i]

    # Plotta la corsia occupata. Usiamo 'o' come marcatore per ogni punto.
    plt.plot(x_values, lanes_extended, marker='o', linestyle='-', markersize=4, label="Corsia occupata da t_0")

    # Imposta i tick sull'asse Y per i numeri interi delle corsie, se possibile.
    # Evita errori se non ci sono dati validi.
    valid_lanes = [l for l in lanes_extended if not np.isnan(l)]
    if valid_lanes:
        min_lane = int(min(valid_lanes))
        max_lane = int(max(valid_lanes))
        plt.yticks(range(min_lane, max_lane + 1))
    else:
        # Se non ci sono dati validi, imposta un range di default o lascia che matplotlib decida.
        plt.yticks(range(0, 5)) # Esempio di range di default

    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Grafico salvato in: {filepath}")


def defineScenario(dbase, vehicleID):
    """Definisce i parametri di uno scenario per un dato veicolo."""
    dbase[vehicleID] = {'nPassenger': random.randint(0, 5),
                        'probPassenger': random.uniform(0, 1),
                        'nPedestrian': random.randint(0, 5),
                        'probPedestrian': random.uniform(0, 1),
                        'altruism': random.uniform(0, 1),
                        }
    return dbase[vehicleID]['nPedestrian']


def calculate_and_store_distance(vehicle_id, person_id, step):
    """
    Calcola la distanza tra un veicolo e un pedone, la memorizza
    e memorizza un flag se la distanza è inferiore a 10 metri.
    """
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
            # DEBUG: Stampa quando un dato di distanza per t_0 viene aggiunto
            print(f"DEBUG: Distanza per t_0 e {person_id} al passo {step}: {distance:.2f}m (Vicino: {is_close})")
        elif vehicle_id == "t_1":
            if pair_key not in distance_data_t1:
                distance_data_t1[pair_key] = []
            distance_data_t1[pair_key].append((distance, is_close))

        # Aggiungi il passo di simulazione solo se non è già presente
        if step not in simulation_steps_unique:
            simulation_steps_unique.append(step)
            simulation_steps_unique.sort() # Mantiene i passi ordinati

    except traci.exceptions.TraCIException as e:
        # print(f"Errore TraCI durante il calcolo della distanza: {e}") # Debugging
        pass # Ignora errori se veicolo/pedone non è più presente


def modify_pedestrian_routes(route_file, num_pedestrians):
    """Modifica il file di route dei pedoni in base al numero specificato."""
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()

        # Rimuovi tutti i pedoni esistenti
        for person in root.findall('person'):
            root.remove(person)

        # Aggiungi nuovi pedoni
        for i in range(num_pedestrians):
            person = ET.SubElement(root, "person", id=f"p_{i}", depart="0.00", type="fast_pedestrian")
            person_trip = ET.SubElement(person, "personTrip", attrib={"from": "-E4", "to": "E6"})

        ET.indent(tree, space="    ", level=0) # Formatta l'XML per leggibilità
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

# Rimuovi e ricrea la directory per grafici puliti ad ogni esecuzione
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
# --- Fine Directory dei grafici ---


# Dati per i grafici
distance_data_t0 = {}
distance_data_t1 = {}
pair_names = {}
simulation_steps_unique = [] # Lista per tenere traccia dei passi di simulazione unici
lane_data_t0 = [] # Nuova lista per memorizzare l'indice della corsia di t_0 ad ogni passo

lista_attivi = [] # Questa variabile non sembra essere usata, può essere rimossa se non necessaria
step_count = 0
simulation_time = 0
max_simulation_time = 3600

# --- Genera lo scenario UNA SOLA VOLTA e ottieni il numero di pedoni ---
dbase = {}
num_pedestrians_scenario = defineScenario(dbase, "t_1")
# DEBUG: Stampa il numero di pedoni generati
print(f"DEBUG: Numero di pedoni generati nello scenario: {num_pedestrians_scenario}")

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


# Inizializza SUMO con traci.start (sempre in modalità automatica)
sumo_cfg_file = os.path.join(script_dir, "TestCreazioneRete", "trolleyNet.sumocfg")

sumoCmd = [
    sumolib.checkBinary('sumo'), # Usa sumo per la modalità automatica
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

        traci.simulationStep()
        step_count += 1
        simulation_time = traci.simulation.getTime()

        # Aggiungi il passo corrente alla lista dei passi unici (se non già presente)
        if step_count not in simulation_steps_unique:
            simulation_steps_unique.append(step_count)
            simulation_steps_unique.sort() # Mantiene ordinata la lista dei passi

        # --- Registra l'occupazione della corsia per il veicolo t_0 ---
        if "t_0" in traci.vehicle.getIDList():
            try:
                lane_index = traci.vehicle.getLaneIndex("t_0")
                lane_data_t0.append(lane_index)
            except traci.exceptions.TraCIException:
                # Se t_0 è presente nell'ID list ma non ha una posizione valida (es. appena partito/arrivato)
                lane_data_t0.append(np.nan) # Aggiungi NaN per indicare un dato mancante
        else:
            # Se t_0 non è proprio presente nella simulazione a questo passo
            lane_data_t0.append(np.nan) # Aggiungi NaN per mantenere la lunghezza allineata ai passi
        # --- Fine registrazione occupazione corsia ---

        # DEBUG: Stampa i veicoli e i pedoni attivi a ogni passo
        current_vehicles = traci.vehicle.getIDList()
        current_persons = traci.person.getIDList()
        # print(f"DEBUG: Passo {step_count}: Veicoli attivi: {current_vehicles}, Pedoni attivi: {current_persons}")


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
                    pass # Ignora errori se la corsia non è valida o il veicolo non è più lì

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
else:
    print("DEBUG: distance_data_t0 è vuoto. Il grafico delle distanze per t_0 non verrà generato.")

if distance_data_t1:
    plot_distances(distance_data_t1, pair_names, simulation_steps_unique, "Distanza Pedoni - Auto t_1 (OSTACOLO)", save_path)

if lane_data_t0:
    plot_lane_occupancy(lane_data_t0, simulation_steps_unique, "Occupazione Corsia Auto t_0 (AV)", save_path)
else:
    print("DEBUG: lane_data_t0 è vuoto. Il grafico dell'occupazione della corsia per t_0 non verrà generato.")

print("Script completato.")