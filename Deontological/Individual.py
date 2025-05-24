import os
import random
import pandas as pd
import numpy as np
import time
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, InputLayer, Input
from keras import backend as K
from Configuration import Configuration # Assicurati che Configuration.py sia accessibile
from sklearn.preprocessing import StandardScaler
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import shutil  # Import the shutil module


random.seed(42)

class Individual:
	"""
	Define an Individual
	Metodo costruttore
	"""
	def __init__(self,conf):
		self.nn = make_nn_individual() # Crea la rete neurale per l'individuo
		self.fitness = 0 # Fitness iniziale
		self.predAction = -1 # Azione predetta (-1: non ancora calcolata, 0: dritto, 1: svolta)
		self.scenario = None # Scenario corrente affrontato dall'individuo

		if conf.randomizeAltruism:
			self.altruism = random.random()
		else:
			self.altruism = conf.ALTRUISM
				
		self.knob = random.random()

	def computeFitness(self, scenario, conf, scaler):
		"""
		Calcola la fitness dell'individuo basata sullo scenario corrente.
		L'auto andrà SEMPRE DRITTA o SVOLTERÀ SEMPRE (approccio deontologico).
		"""
		
		# --- DEONTOLOGICAL APPROACH ---
		# L'auto viene addestrata per andare sempre dritta o svoltare sempre
		# predAction = 0 significa andare dritto e salvare i passeggeri
		# predAction = 1 significherebbe svoltare e salvare i pedoni

		nn_input = scenario[0].reshape(1, -1)

		self.predAction = 0

		if self.nn.predict(nn_input)[0][0] > 0.5: 
			self.predAction = 1
		else:
			self.predAction = 0
		
		#print("SAVE PASSENGERS:", str(conf.savePassengers))
		#print("Predicted Action:", str(self.predAction))
		return self.predAction # Ritorna l'azione (sempre 0 o sempre 1)


	def computeSelfEsteem(self, conf):
		"""
		Aggiorna la fitness dell'individuo basandosi sui reward
		Questo dipende da come l'azione dell'individuo si confronta
		con la regola imposta (sempre dritto o svolta sempre)
		"""

		reward = 0 # Ricompensa o penalità
	
		# Devo salvare i pedoni --> predAction = 1
		if (conf.savePassengers == 0):
			if (self.predAction == 1):
				reward = 0.25
			else:
				reward = -0.25

		# Devo salvare i passeggeri --> predAction = 0
		if (conf.savePassengers == 1):
			if (self.predAction == 0):
				reward = 0.25
			else:
				reward = -0.25

		self.fitness += reward
		# Assicura che la fitness rimanga nel range [0,1] anche dopo i reward
		self.fitness = max(0, min(1, self.fitness))


def make_nn_individual():
	"""
	Crea e compila un modello di rete neurale sequenziale per un individuo.
	Input: 5 caratteristiche dello scenario.
	Output: 1 valore (il 'knob').
	"""
	m_model = Sequential()
	m_model.add(Dense(3, input_dim=5, activation='relu')) # Strato nascosto con 3 neuroni e attivazione ReLU
	# m_model.add(Dense(5,  activation='relu')) # Esempio di un altro strato nascosto (opzionale)
	m_model.add(Dense(1, activation='sigmoid')) # Strato di output con 1 neurone e attivazione sigmoide per output tra 0 e 1

	# Compila il modello
	# 'adam' è un ottimizzatore comune.
	# 'mse' (mean squared error) è una funzione di loss adatta per problemi di regressione (come predire 'knob').
	m_model.compile(optimizer='adam', loss='mse')
	
	# m_model.summary() # Decommenta per stampare un riassunto del modello
	return m_model

def generate_first_population_randomly(conf):
	"""
	Crea una popolazione iniziale di individui in modo casuale.
	:param conf: Oggetto di configurazione contenente POPULATION_SIZE.
	:return: Lista di individui generati.
	"""
	print("[+] Creating Initial NN Model Population Randomly: ", end='')
	result = []
	run_start = time.time()
	for _ in range(conf.POPULATION_SIZE): # Usa _ se l'indice non è necessario
		temp_individual = Individual(conf)
		result.append(temp_individual)
	run_stop = time.time()
	print(f"Done > Takes {run_stop-run_start:.2f} sec")
	return result

def mutate_chromosome(conf, individual=None):
	"""
	Applica una mutazione casuale ai pesi della rete neurale dell'individuo.
	:param conf: Oggetto di configurazione con i parametri di mutazione.
	:param individual: L'individuo da mutare.
	:return: L'individuo mutato.
	"""
	if individual is None:
		return None

	for layer in individual.nn.layers:
		original_weights = layer.get_weights() # Lista di array NumPy (pesi e bias)
		mutated_weights_list = []

		for weights_array in original_weights:
			mutated_array = np.copy(weights_array) # Lavora su una copia
			# Itera su ogni elemento dell'array dei pesi/bias
			for index in np.ndindex(weights_array.shape):
				if random.random() < conf.HIDDEN_LAYER_MUTATION_PROBABILITY:
					# Calcola la mutazione
					current_value = weights_array[index]
					# Assicura che HIDDEN_LAYER_MUTATION_RANGE sia un float
					mutation_range = float(conf.HIDDEN_LAYER_MUTATION_RANGE) if hasattr(conf, 'HIDDEN_LAYER_MUTATION_RANGE') else 0.01
					delta = current_value * mutation_range 
					delta = random.uniform(-abs(delta), abs(delta)) # Assicura che delta sia simmetrico
					mutated_array[index] += delta
			mutated_weights_list.append(mutated_array)
		layer.set_weights(mutated_weights_list)
	return individual


def generate_children(conf, mother: Individual, father: Individual) -> Individual:
	"""
	Genera un nuovo individuo (figlio) combinando i genomi (pesi della NN)
	della madre e del padre.
	:param conf: Oggetto di configurazione.
	:param mother: Individuo madre.
	:param father: Individuo padre.
	:return: Un nuovo individuo figlio.
	"""
	children = Individual(conf) # Crea un nuovo individuo con una NN inizializzata

	# Itera attraverso gli strati della rete neurale
	for layer_idx in range(len(children.nn.layers)):
		child_layer = children.nn.layers[layer_idx]
		mother_layer = mother.nn.layers[layer_idx]
		father_layer = father.nn.layers[layer_idx]

		mother_weights_list = mother_layer.get_weights() # Lista di [pesi, bias]
		father_weights_list = father_layer.get_weights()
		child_new_weights_list = []

		# Itera attraverso le matrici di pesi e i vettori di bias per lo strato corrente
		for mother_w_matrix, father_w_matrix in zip(mother_weights_list, father_weights_list):
			child_w_matrix = np.copy(mother_w_matrix) # Inizia con una copia (es. dalla madre)
			
			# Itera su ogni gene (peso/bias) nella matrice/vettore
			for index in np.ndindex(mother_w_matrix.shape):
				if random.randint(0, 1) == 0: # Scegli casualmente il gene dalla madre o dal padre
					child_w_matrix[index] = mother_w_matrix[index]
				else:
					child_w_matrix[index] = father_w_matrix[index]
			child_new_weights_list.append(child_w_matrix)
		
		child_layer.set_weights(child_new_weights_list) # Imposta i nuovi pesi per lo strato del figlio
	return children

def tournament_selection(population, selectionSize):
	"""
	Esegue la selezione a torneo sulla popolazione.
	Ogni torneo seleziona 1 vincitore tra 2 individui scelti casualmente.
	:param population: Lista di Individui.
	:param selectionSize: Numero di individui da selezionare.
	:return: Lista degli Individui selezionati.
	"""
	if not population: # Se la popolazione è vuota, ritorna una lista vuota
		return []
	if selectionSize <= 0:
		return []

	selected_parents = []
	population_size = len(population)
	
	if population_size == 0: return []


	for _ in range(selectionSize):
		# Scegli due individui a caso per il torneo
		# Assicurati che ci siano almeno due individui per scegliere tra diversi, se possibile
		if population_size == 1:
			tournament_parent1 = population[0]
			tournament_parent2 = population[0] # Se c'è un solo individuo, compete contro se stesso
		else:
			idx1, idx2 = random.sample(range(population_size), 2)
			tournament_parent1 = population[idx1]
			tournament_parent2 = population[idx2]

		# Il vincitore è quello con la fitness maggiore
		if tournament_parent1.fitness >= tournament_parent2.fitness:
			selected_parents.append(tournament_parent1)
		else:
			selected_parents.append(tournament_parent2)
			
	return selected_parents
				
def evolve_population(population, conf, crossover=True, elite=0):
	"""
	Evolve e crea la prossima generazione di individui.
	:param population: Popolazione corrente.
	:param conf: Oggetto di configurazione.
	:param crossover: Booleano, se applicare il crossover.
	:param elite: 0 per selezione a torneo, 1 per elitismo.
	:return: Una nuova popolazione.
	"""
	parents = []
	if not population: # Se la popolazione di input è vuota, ritorna una lista vuota
		return []

	if elite == 1:
		# Ordina i candidati per fitness (dal migliore al peggiore)
		population.sort(key=lambda x: x.fitness, reverse=True)
		
		# Seleziona i migliori N candidati e Y candidati casuali
		# Assicurati che BEST_CANDIDATES_COUNT non superi la dimensione della popolazione
		num_best_to_select = min(conf.BEST_CANDIDATES_COUNT, len(population))
		parents.extend(population[0:num_best_to_select])
		
		# Aggiungi candidati casuali, assicurandoti di non superare la dimensione della popolazione
		# e che ci siano individui tra cui scegliere.
		if population: # Solo se la popolazione non è vuota
			for _ in range(conf.RANDOM_CANDIDATES_COUNT):
				if len(parents) < conf.POPULATION_SIZE : # Non aggiungere più genitori della dimensione della popolazione
					parents.append(population[random.randint(0, len(population) - 1)])
	else: # Selezione a torneo
		num_to_select_tournament = min(conf.BEST_CANDIDATES_COUNT, len(population)) # Quanti genitori selezionare
		if num_to_select_tournament > 0:
			parents = tournament_selection(population, num_to_select_tournament)
	
	if not parents: # Se nessun genitore è stato selezionato (es. popolazione iniziale molto piccola)
		# Riempi la nuova popolazione con individui casuali fino a POPULATION_SIZE
		new_population = []
		print("Warning: No parents selected, filling new population with random individuals.")
		for _ in range(conf.POPULATION_SIZE):
			new_population.append(Individual(conf))
		return new_population

	new_population = []
	if crossover == False: # Se non c'è crossover, i genitori diventano la nuova popolazione (potrebbe necessitare di clonazione)
		# Per evitare di modificare i genitori originali, clonali se necessario.
		# E riempi fino a POPULATION_SIZE con mutazioni dei genitori.
		temp_pop = [copy.deepcopy(p) for p in parents]
		while len(new_population) < conf.POPULATION_SIZE:
			if not temp_pop: break # Evita loop infinito se temp_pop è vuota
			parent_to_mutate = random.choice(temp_pop)
			new_population.append(mutate_chromosome(conf, copy.deepcopy(parent_to_mutate)))
		if not new_population and conf.POPULATION_SIZE > 0: # Fallback se new_population è ancora vuota
			for _ in range(conf.POPULATION_SIZE): new_population.append(Individual(conf))
		return new_population[:conf.POPULATION_SIZE]


	# Creazione della nuova popolazione tramite crossover e mutazione
	# Aggiungi i genitori d'élite direttamente alla nuova popolazione per preservarli (elitismo forte)
	if elite == 1 and conf.BEST_CANDIDATES_COUNT > 0:
	    # Assicurati di aggiungere copie per evitare modifiche inaspettate
		elite_individuals = [copy.deepcopy(p) for p in population[0:min(conf.BEST_CANDIDATES_COUNT, len(population))]]
		new_population.extend(elite_individuals)


	while len(new_population) < conf.POPULATION_SIZE:
		# Seleziona due genitori dalla lista dei 'parents'
		if len(parents) == 1: # Caso speciale: un solo genitore disponibile
			parent_a = parents[0]
			parent_b = parents[0] # Il genitore si accoppia con se stesso (o meglio, una sua copia mutata)
		else:
			parent_a, parent_b = random.sample(parents, 2) # Scegli 2 genitori distinti se possibile

		# Genera un figlio e applica la mutazione
		child = generate_children(conf, mother=parent_a, father=parent_b)
		mutated_child = mutate_chromosome(conf, child)
		new_population.append(mutated_child)

	return new_population[:conf.POPULATION_SIZE] # Assicura che la popolazione non superi la dimensione definita

# La funzione fastCreateScenarios sembra una duplicazione/alternativa di createScenarios.
# Assicurati di usare quella corretta o di unificarle. Per ora, la lascio commentata
# se createScenarios è quella principale usata in ga_general.py.
# def fastCreateScenarios(conf, population, randomize=True): ...

def createScenarios(conf, population, randomize=True):
	"""
	Crea un DataFrame di scenari per la popolazione data.
	Ogni riga del DataFrame è uno scenario [nPed, probPed, nPass, probPass, AltruismLevel].
	:param conf: Oggetto di configurazione.
	:param population: Lista di individui (usata per l'altruismo se non randomizzato per scenario).
	:param randomize: Se True, le probabilità di morte sono casuali per ogni scenario.
	:return: DataFrame pandas con gli scenari.
	"""
	scenarios_list = [] # Lista per raccogliere i dati degli scenari

	for p_individual in population:
		# Numero di pedoni e passeggeri (varia da 1 a N+1)
		num_ped = random.randint(1, conf.numberOfPedestrians + 1) 
		num_pass = random.randint(1, conf.numberOfPassengers + 1)

		# Probabilità di morte per pedoni e passeggeri
		if randomize: # Se True, le probabilità sono casuali per questo scenario
			prob_death_ped = random.random()
			prob_death_pass = random.random()
		else: # Altrimenti, usa i valori fissi dalla configurazione
			prob_death_ped = conf.probDeathPedestrians
			prob_death_pass = conf.probDeathPassengers

		# Altruismo: usa quello dell'individuo
		altruism_level = p_individual.altruism

		scenario_data = [num_ped, prob_death_ped, num_pass, prob_death_pass, altruism_level]
		scenarios_list.append(scenario_data)
	
	df_scenarios = pd.DataFrame(scenarios_list, columns=["numberOfPedestrians",
														 "probPed", 
														 "numberOfPassengers",
														 "probPass",
														 "AltruismLevel"])
	return df_scenarios

def standardize(scenari_df):
	"""
	Standardizza le feature del DataFrame degli scenari usando StandardScaler.
	:param scenari_df: DataFrame pandas con gli scenari.
	:return: Tuple (array NumPy degli scenari standardizzati, oggetto scaler fittato).
	"""
	scaler = StandardScaler()
	# Adatta lo scaler solo se ci sono dati (almeno una riga)
	if not scenari_df.empty:
		# Applica fit_transform che adatta lo scaler e poi trasforma i dati
		standardized_data = scaler.fit_transform(scenari_df)
		return standardized_data, scaler
	# Se il DataFrame è vuoto, ritorna un array vuoto e lo scaler non fittato
	return np.array([]).reshape(0, scenari_df.shape[1] if scenari_df.shape[1] > 0 else 5), scaler


def save_results(df, conf, gen=0):
	"""
	Salva il DataFrame dei risultati di una generazione in un file CSV.
	:param df: DataFrame da salvare.
	:param conf: Oggetto di configurazione (per il percorso di salvataggio).
	:param gen: Numero della generazione corrente.
	"""
	s = "%03d" % gen # Formatta il numero della generazione (es. 001, 010, 100)
	# Usa os.path.join per creare percorsi in modo compatibile con diversi OS
	pathLog = os.path.join(conf.path, "logs", "0") # La sottocartella "0" sembra fissa
	
	try:
		os.makedirs(pathLog, exist_ok=True) # Crea la directory se non esiste (exist_ok=True evita errori se esiste già)
	except OSError as e:
		print(f"Error creating directory {pathLog}: {e}")
		return # Esce se non può creare la directory
	
	file_path = os.path.join(pathLog, f"gen_{s}.txt")
	try:
		df.to_csv(file_path, sep="\t", decimal=",", index=False) # index=False per non scrivere l'indice del DataFrame nel CSV
		# print(f"Results for generation {gen} saved to {file_path}")
	except IOError as e:
		print(f"Error saving results to {file_path}: {e}")
	
def save_accuracy(df, conf, file_name=None, gen=0):
	"""
	Salva il DataFrame delle metriche di accuratezza in un file CSV.
	:param df: DataFrame da salvare.
	:param conf: Oggetto di configurazione.
	:param file_name: Nome del file (se None, ne viene generato uno basato sulla generazione).
	:param gen: Numero della generazione (usato se file_name è None).
	"""
	s = "%03d" % gen
	pathLog = os.path.join(conf.path, "logs", "0")
	
	try:
		os.makedirs(pathLog, exist_ok=True)
	except OSError as e:
		print(f"Error creating directory {pathLog}: {e}")
		return

	actual_file_name = f"accuracy_{s}.txt" if file_name is None else file_name
	file_path = os.path.join(pathLog, actual_file_name)
	try:
		df.to_csv(file_path, sep="\t", decimal=",", index=False)
		# print(f"Accuracy data saved to {file_path}")
	except IOError as e:
		print(f"Error saving accuracy to {file_path}: {e}")


def save_options(conf):
	"""
	Salva i parametri di configurazione in un file di testo.
	:param conf: Oggetto di configurazione.
	"""
	try:
		os.makedirs(conf.path, exist_ok=True) # Crea la directory base del path se non esiste
	except OSError as e:
		print (f"Error creating directory {conf.path}: {e}")
		return
		
	file_path = os.path.join(conf.path, "out.txt")
	try:
		# Usa 'with open' per assicurare che il file venga chiuso correttamente anche in caso di errori
		with open(file_path, "w") as f:
			f.write("Experiment Configuration Options:\n")
			f.write("="*30 + "\n")
			for attribute, value in conf.__dict__.items():
				f.write(f"{attribute}: {str(value)}\n")
		# print(f"Configuration options saved to {file_path}")
	except IOError as e:
		print(f"Error saving options to {file_path}: {e}")