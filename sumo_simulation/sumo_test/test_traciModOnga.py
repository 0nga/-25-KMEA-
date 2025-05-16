import os
import sys
import random
import time
from keras.models import load_model
import traci
import sumolib
import matplotlib.pyplot as plt
import numpy as np

# Carico Rete neurale
try:
    model_name = "sumo_simulation/sumo_test/final_generation_models/individual_0.keras"
    model = load_model(model_name)
    print("Modello caricato con successo da", model_name)
    model.summary()

except Exception as e:
    print(f"Si è verificato un errore durante il caricamento del modello: {e}")
    model = None  # Imposta model a None per gestire il caso di errore

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'share/sumo/tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def removeID(lista, x):
    if (x != 0) and (x in lista):
        lista.remove(x)
    return lista

def plot_distances(distance_data, pair_names, simulation_steps, title):
    """
    Genera un grafico che mostra l'andamento delle distanze nel tempo per ogni coppia veicolo-pedone.
    Args:
        distance_data (dict): Dizionario contenente i dati delle distanze.
        pair_names (dict): Dizionario contenente i nomi dei veicoli e pedoni per ogni coppia.
        simulation_steps (list): Lista dei passi di simulazione.
        title (str): Titolo del grafico.
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Tempo (passi di simulazione)")
    plt.ylabel("Distanza (metri)")
    plt.title(title)
    plt.grid(True)

    if not distance_data:
        print("Nessun dato di distanza da graficare.")
        return

    # Trova il numero massimo di passi di simulazione per uniformare l'asse x
    max_steps = len(simulation_steps)
    x_values = range(1, max_steps + 1)  # Crea una lista di valori x da 1 al numero massimo di passi

    # Itera sulle coppie veicolo-pedone
    for pair_key, distances in distance_data.items():
        vehicle_name, person_name = pair_names[pair_key]
        # Estendi la lista delle distanze con l'ultimo valore per farla corrispondere alla lunghezza di x_values
        if len(distances) < max_steps:
            last_distance = distances[-1]
            distances_extended = distances + [last_distance] * (max_steps - len(distances))
        else:
            distances_extended = distances
        plt.plot(x_values, distances_extended, label=f"Pedone: {person_name}")  # Modificato: solo pedone nella legenda

    plt.legend()
    plt.show()


def defineScenario(dbase, vehicleID):
    dbase[vehicleID] = {'nPassenger': random.randint(0, 5)}


# Inizializza la connessione SUMO
t = traci.connect(27910)

# Dizionari per memorizzare i dati delle distanze nel tempo, separati per veicolo
distance_data_t0 = {}
distance_data_t1 = {}

# Dizionario per memorizzare i nomi dei veicoli e pedoni coinvolti nel calcolo della distanza
pair_names = {}

# Contatore per il passo di simulazione, utile per l'asse x del grafico
simulation_steps = []

def calculate_and_store_distance(vehicle_id, person_id, step):
    """
    Calcola la distanza tra un veicolo e un pedone, la memorizza insieme al tempo,
    e gestisce i dati per il grafico.

    Args:
        vehicle_id (str): L'ID del veicolo.
        person_id (str): L'ID del pedone.
        step (int): Il passo di simulazione corrente.
    """
    global distance_data_t0, distance_data_t1, pair_names, simulation_steps  # Modificato per i dizionari separati
    posV = t.vehicle.getPosition(vehicle_id)
    posP = t.person.getPosition(person_id)
    distance = t.simulation.getDistance2D(posV[0], posV[1], posP[0], posP[1], isDriving=True)
    # minGap = t.vehicle.getMinGap(vehicle_id) #Non usato

    # Crea una chiave univoca per la coppia veicolo-pedone
    pair_key = f"{vehicle_id}-{person_id}"
    if pair_key not in pair_names:
        pair_names[pair_key] = (vehicle_id, person_id)  # Salva i nomi della coppia

    # Inizializza la lista delle distanze per questa coppia, se non esiste
    if vehicle_id == "t_0":
        if pair_key not in distance_data_t0:
            distance_data_t0[pair_key] = []
        distance_data_t0[pair_key].append(distance)
    elif vehicle_id == "t_1":
        if pair_key not in distance_data_t1:
            distance_data_t1[pair_key] = []
        distance_data_t1[pair_key].append(distance)

    simulation_steps.append(step)  # Salva il passo della simulazione

lista_attivi = []
step_count = 0  # Inizializza il contatore
simulation_time = 0  # Inizializza il tempo di simulazione
max_simulation_time = 10  # Definisci la durata massima della simulazione in secondi

while simulation_time < max_simulation_time:

    l = t.simulationStep()
    step_count += 1  # Incrementa il contatore
    simulation_time = t.simulation.getTime()  # Ottieni il tempo di simulazione corrente
    collisions = t.simulation.getCollisions()

    for v1 in t.vehicle.getIDList():
        for p1 in t.person.getIDList():
            calculate_and_store_distance(v1, p1, step_count)  # Calcola e memorizza la distanza

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

# Chiama la funzione per generare i grafici dopo la chiusura della simulazione
if distance_data_t0:
    plot_distances(distance_data_t0, pair_names, simulation_steps, "Distanza Pedoni - Auto t_0")
if distance_data_t1:
    plot_distances(distance_data_t1, pair_names, simulation_steps, "Distanza Pedoni - Auto t_1")

t.close()



