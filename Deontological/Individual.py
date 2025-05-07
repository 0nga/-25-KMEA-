import os
import random
#import gym
import pandas as pd
import numpy as np
import time
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, InputLayer, Input
from keras import backend as K
from Configuration import Configuration
from sklearn.preprocessing import StandardScaler
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys, getopt

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
		self.predAction = -1
		self.scenario = None

		if conf.randomizeAltruism:
			self.altruism = random.random()
		else:
			self.altruism = conf.ALTRUISM

		self.knob = random.random()
		#print("Knob initialization",self.knob)

	def computeFitness(self, scenario, conf, scaler, avgKnobLevel=0):
		self.scenario = copy.deepcopy(scenario)
		# Standardization of the input for NN
		scenario_copia=copy.deepcopy(scenario)
		self.knob = self.nn.predict(scenario_copia)[0][0]
		if self.knob > 1: self.knob = 1
		if self.knob < 0: self.knob = 0

		scenario = scaler.inverse_transform(scenario)
		
		#Evaluate scenario
		evaluatePedestrian = scenario[0][0] * scenario[0][1]
		evaluatePassengers = scenario[0][2] * scenario[0][3]
		
		#predAction=1 means turn, predAction=0 means go straight
		self.predAction = 0
		if (self.knob * evaluatePassengers < evaluatePedestrian * (1-self.knob)):
			self.predAction = 1
			
		# Deontological rules:
		# 1. Never intentionally harm pedestrians (they have right of way)
		# 2. Never intentionally harm passengers (they are in your care)
		# 3. Follow traffic rules and maintain vehicle control
		# 4. Minimize risk to all parties
		
		# Base fitness starts at 0
		utility = 0
		
		# Rule 1: Penalize if choosing to harm pedestrians when they have right of way
		if self.predAction == 0 and scenario[0][0] > 0:  # Going straight when pedestrians present
			utility -= 1.0  # Strong penalty for violating pedestrian right of way
			
		# Rule 2: Penalize if choosing to harm passengers
		if self.predAction == 1 and scenario[0][2] > 0:  # Turning when passengers present
			utility -= 0.8  # Penalty for harming passengers
			
		# Rule 3: Reward for following traffic rules
		if self.predAction == 0 and scenario[0][0] == 0:  # Going straight when no pedestrians
			utility += 0.5  # Reward for following rules
		elif self.predAction == 1 and scenario[0][2] == 0:  # Turning when no passengers
			utility += 0.5  # Reward for following rules
			
		# Rule 4: Reward for minimizing risk
		if scenario[0][1] < 0.5 and scenario[0][3] < 0.5:  # Low probability of harm
			utility += 0.3  # Reward for safe driving
			
		# Normalize fitness to 0-1 range
		utility_max = 1.0  # Maximum possible utility
		utility_min = -2.0  # Minimum possible utility
		self.fitness = (utility - utility_min) / (utility_max - utility_min)
					
		return self.predAction

	def computeSelfEsteem(self, conf, avgKnobLevel):
		'''
		Compute the action performed by an average individual
		'''
		avgAction=0
		#Evaluate scenario
		evaluatePedestrian = self.scenario[0][0] * self.scenario[0][1]
		evaluatePassengers = self.scenario[0][2] * self.scenario[0][3]
		#print(f"avgKnobLevel: {avgKnobLevel}")
		if(avgKnobLevel * evaluatePassengers < evaluatePedestrian * (1-avgKnobLevel)):
			avgAction = 1
			
		reward=0
			
		'''
		 If on average community choose a selfish action and I choose an altruistic action then the reward is honorable and weighted with my altruistic behaviour 
		 ''' 
		if(avgAction == 0 and self.predAction == 1):
			reward = conf.HONOR #*altruism

		'''
		 If on average community choose an altruistic action and I choose a selfish action then the reward is a stigma weighted with my altruistic behaviour 
		 '''
		if(avgAction == 1 and self.predAction == 0):
			reward = conf.STIGMA #* selfish

		self.fitness += reward

def make_nn_individual():
	m_model = Sequential()

	'''inp = Input(shape=(6,1))
	x = Dense(3, activation='relu', name='dense1')(inp)
	out = Dense(1, activation='tanh', name='dense2')(x)
	m_model=Model(inp, out)'''
	
	m_model.add(Dense(3, input_dim=5))
	#m_model.add(Dense(5,  activation='relu')) prova anche questo
	#m_model.add(Dense(1,  activation='tanh'))
	m_model.add(Dense(1))

	# Compile Neural Network
	m_model.compile(optimizer='adam', loss='categorical_crossentropy')
	# provare con loss='mse'
	
	'''
	CHATGPT
	Problemi nel codice:
	loss='categorical_crossentropy':
	Questo tipo di loss è usato per classificazione multi-classe con one-hot encoding.
	Ma il modello ha un solo output Dense(1), quindi probabilmente serve un'altra loss function.
	'''
	#m_model.summary()
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


# Suggerimento: Se vuoi un comportamento più realistico biologicamente, potresti usare un crossover uniforme 
# con probabilità, o una media pesata (es. interpolazione tra geni). Fammi sapere se vuoi un esempio di quello.
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
	
	'''print("Original")
	print([p.fitness for p in population])
	print("A")
	print([p.fitness for p in population_a])
	print("B")
	print([p.fitness for p in population_b])'''
	
	population_result = []

	for i in range(selectionSize):
		index_a = list_index_a[i]
		index_b = list_index_b[i]
		if population[index_a].fitness > population[index_b].fitness:
			population_result.append(population[index_a])
		else:
			population_result.append(population[index_b])
			
	'''print("result")
	print([p.fitness for p in population_result])		
	sys.exit()'''
	
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

def fastCreateScenarios(conf, population, randomize=True):
    import pandas as pd
    import numpy as np
    import random

    scenarios = []

    for p in population:
        nPed = random.randint(1, conf.numberOfPedestrians + 1)
        nPass = random.randint(1, conf.numberOfPassengers + 1)

        if randomize:
            probDeathPedestrians = random.random()
            probDeathPassengers = random.random()
        else:
            probDeathPedestrians = conf.probDeathPedestrians
            probDeathPassengers = conf.probDeathPassengers

        scenario = [nPed, probDeathPedestrians, nPass, probDeathPassengers, p.altruism]
        scenarios.append(scenario)

    df = pd.DataFrame(scenarios, columns=[
        "numberOfPedestrians",
        "probPed",
        "numberOfPassengers",
        "probPass",
        "AltruismLevel"
    ])
    return df

def createScenarios(conf,population,randomize=True):
	df=pd.DataFrame(columns=["numberOfPedestrians",
							"probPed", 
							"numberOfPassengers",
							"probPass",
							"AltruismLevel"
							])

	for p in population:
			
			#Number of pedestrian varies in 1 - numberOfPedestrians+1
			nPed = random.randint(1,conf.numberOfPedestrians+1) 
			#nPed = 1
			#Number of passengers varies in 1 - numberOfPedestrians+1
			nPass = random.randint(1,conf.numberOfPassengers+1)
			#nPass = 1

			if randomize:
				probDeathPedestrians = random.random()
				probDeathPassengers = random.random()
			else:
				probDeathPedestrians = conf.probDeathPedestrians
				probDeathPassengers = conf.probDeathPassengers

			scenario = [nPed,probDeathPedestrians,nPass,probDeathPassengers,p.altruism]
			#print("Prima",scenario)
			scenario = np.array(scenario)
			scenario = scenario.reshape((1,5))
			#predAction = p.computeFitness(scenario, conf)

			#isSvolta =(l.KnobLevel*l.numberOfPassengers < (1-l.KnobLevel)*prob*l.numberOfPedestrians)
			#isSvolta =(scenario[0][5] * scenario[0][2] < (1-scenario[0][5]) * prob * scenario[0][0])
			#print(scenario)
			#print(scenario.shape)
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
		#print ("Creation of the directory %s failed" % pathLog)
		print()
	else:
		#print ("Successfully created the directory %s" % pathLog)
		print()
	
	df.to_csv(os.path.join(pathLog, "gen_"+s+".txt"), sep="\t", decimal=",")
	
def save_accuracy(df,conf,file_name=None,gen=0):
	
	s="%03d"%gen
	pathLog = conf.path + "/logs/0"
	
	try:
		os.makedirs(pathLog)
	except OSError:
		#print ("Creation of the directory %s failed" % pathLog)
		print()
	else:
		#print ("Successfully created the directory %s" % pathLog)
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
		#print(attribute, '=', value)
		f.write(attribute + ": " + str(value)+"\n")
	
	f.close()
