import os
import random
import pandas as pd
import numpy as np
import time
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, InputLayer, Input
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import copy


import random
random.seed(42)

class Individual:

	"""
	Define an Individual
	Metodo costruttore
	"""
	def __init__(self,conf):
		self.nn = make_nn_individual()

		self.fitness = 0
		self.age = 0
		self.predAction = -1 # Ora sarà determinato dalla NN
		self.scenario = None

		if conf.randomizeAltruism:
			self.altruism = random.random()
		else:
			self.altruism = conf.ALTRUISM

		# self.knob = random.random() # Questo non serve più essere random, sarà l'output della NN

	def computeFitness(self, scenario, conf, scaler):
		self.scenario = copy.deepcopy(scenario)

		# Denormalizza lo scenario per i calcoli utilitaristici
		scenario_denormalized = scaler.inverse_transform(scenario)

		# Prendi gli input per la rete neurale
		# numberOfPedestrians, probPed, numberOfPassengers, probPass, AltruismLevel
		nn_input = scenario[0].reshape(1, -1) # Assicurati che sia nella forma (1, 5) per la predict

		# >>> PASSO FONDAMENTALE: LA RETE NEURALE PRENDE LA DECISIONE <<<
		# L'output della NN sarà il nostro 'knob'. Assicurati che la NN sia configurata con un output adatto,
		# ad esempio una singola unità con attivazione sigmoid per un valore tra 0 e 1, o tanh tra -1 e 1.
		# Per una decisione binaria (svolta/dritto), una sigmoid è spesso utile.
		self.knob = self.nn.predict(nn_input)[0][0] # Prende il primo (e unico) valore di output

		# Basandosi sull'output della rete neurale, prendi la decisione (predAction)
		# Ad esempio, se l'output è > 0.5, svolta, altrimenti vai dritto.
		if self.knob > 0.5: # Esempio di soglia, da calibrare
			self.predAction = 1 # Svolta
		else:
			self.predAction = 0 # Dritto

		# Estrai i valori dello scenario denormalizzati per i calcoli di utilità
		temp_numberOfPedestrians = scenario_denormalized[0][0]
		temp_probDeathPedestrians = scenario_denormalized[0][1]
		temp_numberOfPassengers = scenario_denormalized[0][2]
		temp_probDeathPassengers = scenario_denormalized[0][3]
		temp_altruism = scenario_denormalized[0][4] # L'altruismo dello scenario/individuo

		# Calcola l'utilità basata sulla predAction presa dalla NN
		# Questa è la parte dove definisci la tua funzione di utilità "utilitaristica"
		# L'obiettivo è massimizzare il bene complessivo (o minimizzare il danno).

		utility = 0.0

		if self.predAction == 0: # L'agente decide di andare dritto
			# Costo per i pedoni, beneficio per i passeggeri (o viceversa)
			# Se va dritto, i pedoni sul percorso potrebbero morire, i passeggeri sono salvi (idealmente)
			cost_pedestrians_straight = temp_numberOfPedestrians * temp_probDeathPedestrians
			benefit_passengers_straight = temp_numberOfPassengers * (1 - temp_probDeathPassengers) # Assumendo probDeathPassengers qui sia la probabilità di morire se si va dritto

			utility = (benefit_passengers_straight * (1 - temp_altruism)) - (cost_pedestrians_straight * temp_altruism)

			# Puoi aggiungere qui costi fissi o bonus
			# utility += conf.HONOR if cost_pedestrians_straight == 0 else conf.STIGMA # Esempio: bonus se nessun pedone muore andando dritto

		else: # self.predAction == 1: # L'agente decide di svoltare
			# Se svolta, i passeggeri potrebbero essere a rischio, i pedoni sono salvi (idealmente)
			cost_passengers_turn = temp_numberOfPassengers * temp_probDeathPassengers # Assumendo probDeathPassengers qui sia la probabilità di morire se si svolta
			benefit_pedestrians_turn = temp_numberOfPedestrians * (1 - temp_probDeathPedestrians) # Assumendo probDeathPedestrians qui sia la probabilità di morire se non si svolta

			utility = (benefit_pedestrians_turn * temp_altruism) - (cost_passengers_turn * (1 - temp_altruism))
			# Puoi aggiungere qui costi fissi o bonus
			# utility += conf.HONOR if cost_passengers_turn == 0 else conf.STIGMA # Esempio: bonus se nessun passeggero muore svoltando

		# La normalizzazione della fitness è ancora utile per mantenere i valori in un range gestibile
		# Per calcolare min/max utility, dovresti considerare i casi estremi per entrambe le azioni
		# (andare dritto e svoltare) con i valori min/max di pedoni, passeggeri e probabilità.
		# Questo può essere complesso, un approccio più semplice per iniziare è una normalizzazione euristica,
		# ma per precisione, calcola i veri estremi.

		# Rivediamo la normalizzazione della fitness basata sui valori min/max possibili di utilità
		# Questo deve essere fatto considerando le formule di utilità e i range dei parametri (nPed, nPass, probDeath, altruism)
		# per trovare i veri min e max dell'utilità in uno scenario.

		# Per esempio, i valori massimi e minimi per i termini potrebbero essere:
		max_nPed = conf.numberOfPedestrians + 1
		max_nPass = conf.numberOfPassengers + 1
		min_nPed = 1
		min_nPass = 1

		# Per semplicità, possiamo usare valori fittizi per max_utility e min_utility
		# ma per un modello robusto dovresti calcolarli con attenzione.
		# Un modo per stimare è provare tutte le combinazioni estreme dei parametri di input.
		# Ad esempio:
		# scenario = [nPed,probDeathPedestrians,nPass,probDeathPassengers,p.altruism]
		# max_utility_straight = (max_nPass * (1 - 0)) - (min_nPed * 1 * 1) # Tutti i passeggeri sopravvivono, un solo pedone muore con certezza (altruismo 1)
		# min_utility_straight = (min_nPass * (1 - 1)) - (max_nPed * 1 * 1) # Tutti i passeggeri muoiono, tutti i pedoni muoiono con certezza (altruismo 1)
		# ... e così via per turn

		# Per ora, userò un placeholder, ma è cruciale affinare questo per una corretta normalizzazione.
		# I tuoi calcoli originali di `utility_straight_max_pass` ecc. provavano a fare questo.
		# Dobbiamo adattarli alla nuova formula di utilità.
		# Ad esempio, per la tua nuova utility:
		# Max utility when predAction = 0: (max_nPass * (1-min_probPass)) - (min_nPed * max_probPed * min_altruism)
		# Min utility when predAction = 0: (min_nPass * (1-max_probPass)) - (max_nPed * min_probPed * max_altruism)
		# E analogamente per predAction = 1

		# Per il momento, userò un approccio più semplice per la normalizzazione:
		# Clamping o mapping a un range noto. Se l'utilità può essere negativa e positiva,
		# puoi mapparla a [0, 1] con un'ipotesi sui valori min/max.
		# Questo è un esempio, calcola i veri estremi della tua funzione di utilità:
		hypothetical_min_utility = - (max_nPed + max_nPass) # Stima pessimistica
		hypothetical_max_utility = (max_nPed + max_nPass)  # Stima ottimistica

		range_utility = (hypothetical_max_utility - hypothetical_min_utility)
		if range_utility == 0: # Evita divisione per zero
			self.fitness = 0.5 # Valore neutro
		else:
			self.fitness = (utility - hypothetical_min_utility) / range_utility
			# Assicurati che la fitness sia tra 0 e 1
			self.fitness = max(0.0, min(1.0, self.fitness))

		return self.predAction # Ritorna l'azione decisa dalla NN

	def computeSelfEsteem(self, conf, ideal_utilitarian_action):
		# Qui, ideal_utilitarian_action è la 'convieneSvolta' calcolata in ga_general.py,
		# che rappresenta la scelta utilitaristica ideale per lo scenario.
		# self.predAction è l'azione effettivamente presa dalla rete neurale.

		reward = 0

		# Se l'azione della NN corrisponde alla scelta utilitaristica ideale
		if self.predAction == ideal_utilitarian_action:
			reward = conf.HONOR
		# Se l'azione della NN NON corrisponde alla scelta utilitaristica ideale
		else:
			reward = conf.STIGMA

		# Applica il reward alla fitness
		self.fitness += reward

		# Potresti voler normalizzare o clampare la fitness dopo l'aggiunta del reward
		# per mantenerla in un range ragionevole, es. [0, 1] o [0, molto grande]
		self.fitness = max(0.0, self.fitness) # Assicurati che non diventi negativa se STIGMA è molto grande


def make_nn_individual():
	m_model = Sequential()

	'''inp = Input(shape=(6,1))
	x = Dense(3, activation='relu', name='dense1')(inp)
	out = Dense(1, activation='tanh', name='dense2')(x)
	m_model=Model(inp, out)'''
	
	m_model.add(Dense(3, input_dim=5))
	m_model.add(Dense(1))

	# Compile Neural Network
	m_model.compile(optimizer='adam', loss='mse')
	
	return m_model

def generate_first_population_randomly(conf):
	"""
	Creates an Initial Random Population
	 :param conf: oggetto di configurazione contenente POPULATION_SIZE
    :return: lista di individui generati
	"""

	print("[+] Creating Initial NN Model Population Randomly: ", end='')

	result = []
	run_start = time.time()

	for current in range(conf.POPULATION_SIZE):
		temp_individual = Individual(conf)
		result.append(temp_individual)
		
	run_stop = time.time()
	print(f"Done > Takes {run_stop-run_start} sec")

	return result

def mutate_chromosome(conf,individual=None):
	"""
	Randomnly mutate individual chromosome
	:param population: Current Population
	:return: A new population
	"""
	#Apply mutation to each weight of the NN 
	for l in individual.nn.layers:
		weights = l.get_weights()
		for i in range(len(weights[0])):
			for j in range(len(weights[0][i])):
				
				#Compute random value in the delta interval
				if random.random() < conf.HIDDEN_LAYER_MUTATION_PROBABILITY:
					delta = weights[0][i][j] * conf.HIDDEN_LAYER_MUTATION_RANGE
					delta = random.uniform(-delta, delta)
					weights[0][i][j] += delta
		l.set_weights(weights)
	return individual

def generate_children(conf,mother: Individual, father: Individual) -> Individual:
	"""
	Generate a New Children based Mother and Father Genomes
	:param mother: Mother Individual
	:param father: Father Individual
	:return: A new Children
	"""
	
	children = Individual(conf)

	for index in range(len(children.nn.layers)):
		l_children=children.nn.layers[index]
		l_mother=mother.nn.layers[index]
		l_father=father.nn.layers[index]
		weights_children = l_children.get_weights()
		weights_mother = l_mother.get_weights()
		weights_father = l_father.get_weights()
		for i in range(len(weights_children[0])):
			for j in range(len(weights_children[0][i])):
				#Compute random value in the delta interval
				if random.randint(0, 1) == 0:
					weights_children[0][i][j] = weights_mother[0][i][j]
				else:
					weights_children[0][i][j] = weights_father[0][i][j]
		l_children.set_weights(weights_children)
	return children

def tournament_selection(population,selectionSize):
	"""
	Perform tournament selection on the population.
	Each tournament selects 1 winner out of 2 randomly chosen individuals.

	:param population: List of Individuals
	:param selectionSize: Number of individuals to select
	:return: List of selected Individuals
	"""

	list_index_a = random.sample(range(len(population)), selectionSize)
	list_index_b = random.sample(range(len(population)), selectionSize)

	population_result = []

	for i in range(selectionSize):
		index_a = list_index_a[i]
		index_b = list_index_b[i]
		if population[index_a].fitness > population[index_b].fitness:
			population_result.append(population[index_a])
		else:
			population_result.append(population[index_b])
		
	return population_result
				
def evolve_population(population, conf, crossover=True, elite=0):
	"""
	Evolve and Create the Next Generation of Individuals
	:param population: Current Population
	:return: A new population
	"""

	parents = []
	if elite==1:
		# Sort Candidates by fitness
		population.sort(key=lambda x: x.fitness, reverse=True)
		
		# Select N Best Candidates + Y Random Candidates. Kill the Rest of Chromosomes
		parents.extend(population[0:conf.BEST_CANDIDATES_COUNT])  # N Best Candidates
		for rn in range(conf.RANDOM_CANDIDATES_COUNT):
			parents.append(population[random.randint(0, conf.POPULATION_SIZE - 1)])  # Y Random Candidate
	else:
		parents = tournament_selection(population, conf.BEST_CANDIDATES_COUNT)
	
	if crossover==False:
		# Create New Population Through Crossover
		new_population = []
		new_population.extend(parents)
	else:
		new_population = []

	while len(new_population) < conf.POPULATION_SIZE:
		parent_a = random.randint(0, len(parents) - 1)
		parent_b = random.randint(0, len(parents) - 1)
		while parent_a == parent_b:
			parent_b = random.randint(0, len(parents) - 1)
		
		temp_individual = mutate_chromosome(conf,
                generate_children(conf,
                    mother=parents[parent_a],
                    father=parents[parent_b]
                )
            )
		new_population.append(temp_individual)

	return new_population

def createScenarios(conf,population,randomize=True):
	df=pd.DataFrame(columns=["numberOfPedestrians",
							"probPed", 
							"numberOfPassengers",
							"probPass",
							"AltruismLevel"
							])

	for p in population:
			
			nPed = random.randint(1,conf.numberOfPedestrians+1) 
			nPass = random.randint(1,conf.numberOfPassengers+1)

			if randomize:
				probDeathPedestrians = random.random()
				probDeathPassengers = random.random()
			else:
				probDeathPedestrians = conf.probDeathPedestrians
				probDeathPassengers = conf.probDeathPassengers

			scenario = [nPed,probDeathPedestrians,nPass,probDeathPassengers,p.altruism]

			scenario = np.array(scenario)
			scenario = scenario.reshape((1,5))
			#predAction = p.computeFitness(scenario, conf)

			#isSvolta =(l.KnobLevel*l.numberOfPassengers < (1-l.KnobLevel)*prob*l.numberOfPedestrians)
			#isSvolta =(scenario[0][5] * scenario[0][2] < (1-scenario[0][5]) * prob * scenario[0][0])
	
			df_temp=pd.DataFrame(scenario,columns=["numberOfPedestrians",
					"probPed", 
					"numberOfPassengers",
					"probPass",
					"AltruismLevel"
					])
			df = pd.concat([df, df_temp], ignore_index=True)
	return df

def standardize(scenari):
	scaler = StandardScaler()
	scaler.fit(scenari)

	return scaler.transform(scenari), scaler

def save_results(df,conf,gen=0):
	
	s="%03d"%gen
	pathLog = conf.path + "/logs/0"
	
	try:
		os.makedirs(pathLog)
	except OSError:
		print()
	else:
		print()
	
	df.to_csv(os.path.join(pathLog, "gen_"+s+".txt"), sep="\t", decimal=",")
	
def save_accuracy(df,conf,file_name=None,gen=0):
	
	s="%03d"%gen
	pathLog = conf.path + "/logs/0"
	
	try:
		os.makedirs(pathLog)
	except OSError:
		print()
	else:
		print()
	
	if file_name == None:
		df.to_csv(os.path.join(pathLog, "accuracy_"+s+".txt"), sep="\t", decimal=",")
	else:
		df.to_csv(os.path.join(pathLog, file_name), sep="\t", decimal=",")

def save_options(conf):
	try:
		os.makedirs(conf.path)
	except OSError:
		print ("Creation of the directory %s failed" % conf.path)
	else:
		print ("Successfully created the directory %s" % conf.path)
		
	f = open(os.path.join(conf.path, "out.txt"), "w")
	
	for attribute, value in conf.__dict__.items():
		f.write(attribute + ": " + str(value)+"\n")

	f.close()
