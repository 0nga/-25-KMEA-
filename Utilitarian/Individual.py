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
		nn_input = scenario[0].reshape(1, -1)

		# >>> PASSO FONDAMENTALE: LA RETE NEURALE PRENDE LA DECISIONE <<<
		# L'output della NN sarà il nostro 'knob'.
		self.knob = self.nn.predict(nn_input)[0][0] # Effettuo la predizione e prendo il primo ed unico valore di output

		# Basandosi sull'output della rete neurale, viene presa la decisione (predAction)
		# Se l'output è > 0.5, svolta, altrimenti vai dritto.
		if self.knob > 0.5:
			self.predAction = 1 # Svolta
		else:
			self.predAction = 0 # Dritto

		# Estrai i valori dello scenario denormalizzati per i calcoli di utilità
		numberOfPedestrians = scenario_denormalized[0][0]
		probDeathPedestrians = scenario_denormalized[0][1]
		numberOfPassengers = scenario_denormalized[0][2]
		probDeathPassengers = scenario_denormalized[0][3]
		# temp_altruism = scenario_denormalized[0][4] # L'altruismo dello scenario/individuo

		# Calcola l'utilità basata sulla predAction presa dalla NN
		# Qua devo definire la utility function "utilitaristica" che abbia come obiettivo
		# la massimizzazione del bene complessivo (o minimizzazione del danno).

		utility = 0.0

		if self.predAction == 0: # La NN decide di andare dritto
			# Costo per i pedoni, beneficio per i passeggeri (o viceversa)
			# Se va dritto, i pedoni sul percorso potrebbero morire, i passeggeri sono salvi (idealmente)
			cost_pedestrians_straight = numberOfPedestrians * probDeathPedestrians
			benefit_passengers_straight = numberOfPassengers * (1 - probDeathPassengers) # probDeathPassengers = probabilità di morire se si va dritto

			utility = (benefit_passengers_straight) - (cost_pedestrians_straight)
			#utility = numberOfPassengers - numberOfPedestrians

			
		else: # self.predAction == 1: # L'agente decide di svoltare
			# Se svolta, i passeggeri potrebbero essere a rischio, i pedoni sono salvi (idealmente)
			cost_passengers_turn = numberOfPassengers * probDeathPassengers # Assumendo probDeathPassengers qui sia la probabilità di morire se si svolta
			benefit_pedestrians_turn = numberOfPedestrians * (1 - probDeathPedestrians) # Assumendo probDeathPedestrians qui sia la probabilità di morire se non si svolta

			utility = (benefit_pedestrians_turn) - (cost_passengers_turn)
			#utility = numberOfPedestrians - numberOfPassengers



		# Rivediamo la normalizzazione della fitness basata sui valori min/max possibili di utilità
		# Questo deve essere fatto considerando le formule di utilità e i range dei parametri (nPed, nPass, probDeath, altruism)
		# per trovare i veri min e max dell'utilità in uno scenario.

		# NORMALIZZAZIONE FUNZIONE FITNESS per mantenerla in un range gestibile 
		# Consideriamo i casi estremi(valori max/min per pedoni/passeggeri/probabilità) per entrambe le azioni (0 e 1)
		# ed individuiamo i valori min e max dell'utilità di uno scenario
		max_nPed = conf.numberOfPedestrians + 1
		max_nPass = conf.numberOfPassengers + 1
		min_nPed = 1
		min_nPass = 1

		# Per la tua nuova utility:
		# Max utility when predAction = 0: (max_nPass * (1-min_probPass)) - (min_nPed * max_probPed * min_altruism)
		# Min utility when predAction = 0: (min_nPass * (1-max_probPass)) - (max_nPed * min_probPed * max_altruism)
		# E analogamente per predAction = 1

		# approccio più semplice per la normalizzazione:Clamping o mapping a un range noto. 
		# Se l'utilità può essere negativa e positiva,
		# puoi mapparla a [0, 1] con un'ipotesi sui valori min/max.
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
		# ideal_utilitarian_action è la 'convieneSvolta' calcolata in ga_general.py,
		# rappresenta la scelta utilitaristica ideale per lo scenario.
		# self.predAction è l'azione effettivamente presa dalla rete neurale.

		reward = 0

		# Se l'azione della NN corrisponde alla scelta utilitaristica ideale
		if self.predAction == ideal_utilitarian_action:
			reward = 0.25
		# Se l'azione della NN NON corrisponde alla scelta utilitaristica ideale
		else:
			reward = -0.25

		# Applica il reward alla fitness
		self.fitness += reward

		# Potresti voler normalizzare o clampare la fitness dopo l'aggiunta del reward
		# per mantenerla in un range ragionevole, es. [0, 1] o [0, molto grande]
		self.fitness = max(0.0, self.fitness) # Assicurati che non diventi negativa se STIGMA è molto grande


	# Tentativo ma l'altra funziona meglio
	def computeFitnessSecondVersion(self, scenario, conf, scaler):
		self.scenario = copy.deepcopy(scenario)

        # Denormalize the scenario for utility calculations
		scenario_denormalized = scaler.inverse_transform(scenario)

        # Get inputs for the neural network
        # numberOfPedestrians, probPed, numberOfPassengers, probPass, AltruismLevel
		nn_input = scenario[0].reshape(1, -1)

        # >>> FUNDAMENTAL STEP: NEURAL NETWORK MAKES THE DECISION <<<
        # The NN output will be our 'knob'.
		self.knob = self.nn.predict(nn_input)[0][0] # Make prediction and take the first and only output value

        # Based on the neural network output, the decision is made (predAction)
        # If the output is > 0.5, turn, otherwise go straight.
		if self.knob > 0.5:
			self.predAction = 1 # Turn
		else:
			self.predAction = 0 # Straight

        # Extract denormalized scenario values for utility calculations
        # Assuming scenario_denormalized[0] contains:
        # [num_pedestrians, prob_death_ped_straight, num_passengers, prob_death_pass_straight, altruism_level]
		numberOfPedestrians = scenario_denormalized[0][0]
		probDeathPedestrians_straight = scenario_denormalized[0][1] # Prob if go straight
		numberOfPassengers = scenario_denormalized[0][2]
		probDeathPassengers_straight = scenario_denormalized[0][3] # Prob if go straight
		temp_altruism = scenario_denormalized[0][4] # Altruism of the scenario/individual

        # Define probabilities if turning (these would ideally come from the scenario or a model)
        # For simplicity, assuming turning might have different probabilities or fixed ones.
        # It's CRITICAL that these are defined based on your scenario's physics/logic.
		probDeathPedestrians_turn = 0.0 # Assuming pedestrians are safe if turning
		probDeathPassengers_turn = 0.5 # Example: passengers have 50% chance of death if turning

		utility = 0.0

        # --- Utilitarian Utility Calculation ---
        # Objective: Maximize overall well-being (minimize total harm)
        # Incorporating altruism (if AltruismLevel is from 0 to 1, where 1 is fully altruistic towards pedestrians)
		
		if self.predAction == 0: # NN decides to go straight
			cost_pedestrians = numberOfPedestrians * probDeathPedestrians_straight
			benefit_passengers = numberOfPassengers * (1 - probDeathPassengers_straight)

            # Utility with altruism: higher altruism penalizes pedestrian harm more, values passenger benefit less
			utility = (benefit_passengers) - (cost_pedestrians)

		else: # self.predAction == 1: # Agent decides to turn
			cost_passengers = numberOfPassengers * probDeathPassengers_turn
			benefit_pedestrians = numberOfPedestrians * (1 - probDeathPedestrians_turn)

            # Utility with altruism: higher altruism values pedestrian benefit more, penalizes passenger harm less
			utility = (benefit_pedestrians * temp_altruism) - (cost_passengers * (1 - temp_altruism))

        # --- FITNESS NORMALIZATION ---
        # Calculate the true global min/max utility values based on all possible input ranges
        # These constants should ideally be pre-calculated once or passed via 'conf'.
        # Assuming conf.MAX_PEDESTRIANS and conf.MAX_PASSENGERS represent the max counts.
        # Assuming probabilities range from 0.0 to 1.0 and altruism from 0.0 to 1.0.

        # --- IMPORTANT: Re-calculate these min/max based on your EXACT utility function and input ranges ---
        # Max possible pedestrians/passengers
		max_nPed = conf.numberOfPedestrians + 1 # Assuming conf.numberOfPedestrians is the max index
		max_nPass = conf.numberOfPassengers + 1 # Assuming conf.numberOfPassengers is the max index

        # Example calculation for min/max utility (assuming altruism from 0 to 1)
        # Max_utility_straight: (max_nPass * (1-0)) * (1-0) - (min_nPed * 0 * 0) = max_nPass
        # Min_utility_straight: (min_nPass * (1-1)) * (1-1) - (max_nPed * 1 * 1) = -max_nPed
        # Max_utility_turn:     (max_nPed * (1-0)) * 1 - (min_nPass * 0 * 0) = max_nPed
        # Min_utility_turn:     (min_nPed * (1-1)) * 1 - (max_nPass * 1 * 0) = 0. No, it would be (min_nPed * 0 * 1) - (max_nPass * 1 * (1-0)) = -max_nPass
        #
        # Re-evaluating the bounds for the utility function:
        # utility = (benefit_A * altruism_factor_A) - (cost_B * altruism_factor_B)
        # Where altruism_factor_A and altruism_factor_B are (altruism) or (1-altruism)

        # Let M = max(max_nPed, max_nPass)
        # Theoretical Max Utility (e.g., max benefit for one group, zero cost for other, altruism aligned):
        # Case 1 (straight): (max_nPass * 1) * (1-0) - (min_nPed * 0 * 0) = max_nPass (when altruism=0)
        # Case 2 (turn): (max_nPed * 1) * 1 - (min_nPass * 0 * 0) = max_nPed (when altruism=1)
        # So, true_max_utility = max(max_nPass, max_nPed)

        # Theoretical Min Utility (e.g., max cost for one group, zero benefit for other, altruism aligned):
        # Case 1 (straight): (min_nPass * 0) * (1-1) - (max_nPed * 1 * 1) = -max_nPed (when altruism=1)
        # Case 2 (turn): (min_nPed * 0) * 1 - (max_nPass * 1 * (1-0)) = -max_nPass (when altruism=0)
        # So, true_min_utility = min(-max_nPed, -max_nPass) = -max(max_nPed, max_nPass)
		
		true_max_utility = max(max_nPed, max_nPass)
		true_min_utility = -max(max_nPed, max_nPass) # This is a robust way to calculate min/max if symmetric

		range_utility = (true_max_utility - true_min_utility)

		if range_utility == 0: # Avoid division by zero, though unlikely with proper bounds
			self.fitness = 0.5 # Neutral value
		else:
			self.fitness = (utility - true_min_utility) / range_utility
            # Ensure fitness is clamped between 0 and 1
			self.fitness = max(0.0, min(1.0, self.fitness))
	
		return self.predAction # Return the action decided by the NN

	# Tentativo ma l'altra funziona meglio
	def computeSelfEsteemSecondVersion(self, conf, ideal_utilitarian_action):
        # ideal_utilitarian_action is the 'convieneSvolta' calculated in ga_general.py,
        # representing the ideal utilitarian choice for the scenario.
        # self.predAction is the action actually taken by the neural network.

        # Rewards for matching the ideal utilitarian action
        # Adjust these values so they don't excessively perturb the 0-1 fitness range
        # For instance, small percentage changes or values that keep fitness within 0-1 after addition.
        
		reward = 0
		if self.predAction == ideal_utilitarian_action:
			reward = 0.1 # Smaller positive reward
		else:
			reward = -0.1 # Smaller negative penalty

        # Apply the reward to the fitness
		self.fitness += reward

        # Re-normalize/clamp fitness after applying the reward
        # Assuming initial fitness is [0, 1] and reward is [-0.1, 0.1]
        # Then, fitness can range from -0.1 to 1.1
        # To re-normalize to [0, 1]:
		min_possible_fitness_after_reward = 0.0 + (-0.1) # Min fitness (0) + min reward (-0.1)
		max_possible_fitness_after_reward = 1.0 + 0.1   # Max fitness (1) + max reward (0.1)
		new_range = max_possible_fitness_after_reward - min_possible_fitness_after_reward

        
		if new_range > 0:            
			self.fitness = (self.fitness - min_possible_fitness_after_reward) / new_range
		else:
			self.fitness = 0.5 # Neutral if range is zero

        # Ensure it stays within [0, 1]
		self.fitness = max(0.0, min(1.0, self.fitness))

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
