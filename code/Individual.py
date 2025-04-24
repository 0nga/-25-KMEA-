import os
import random
#import gym
import pandas as pd
import numpy as np
#import tensorflow as tf
import time
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, InputLayer, Input
from keras import backend as K
from Configuration import Configuration
from sklearn.preprocessing import StandardScaler
import copy

import sys, getopt

import random
random.seed(42)

class Individual:

	"""
	Define an Individual
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
		#scenario_copia[0][0]=(scenario_copia[0][0]-3.5)/1.7
		#scenario_copia[0][2]=(scenario_copia[0][2]-3.5)/1.7

		#print("Scenario pre predict ",scenario_copia)
		self.knob = self.nn.predict(scenario_copia)[0][0]
		if self.knob > 1: self.knob = 1
		if self.knob < 0: self.knob = 0

		#print(f"prima {scenario}")
		scenario = scaler.inverse_transform(scenario)
		#print(f"dopo {scenario}")
		#print("Predicction ",self.nn.predict(scenario_copia))
		

		#Evaluate scenario
		evaluatePedestrian = scenario[0][0] * scenario[0][1]
		evaluatePassengers = scenario[0][2] * scenario[0][3]
		# Check whether the cost should be computed or not
		# If the probability of harming pedestrian is greater than the probability of harming passenger than cost should 
		# be computed 
		computeCost = 0
		if (evaluatePedestrian > evaluatePassengers):
			computeCost = 1
		
		#predAction=1 means turn, predAction=0 means go straight
		self.predAction = 0
		if (self.knob * evaluatePassengers < evaluatePedestrian * (1-self.knob)):
			self.predAction = 1
			
		selfish = 1 - self.altruism
		
		temp_numberOfPedestrians = scenario[0][0]
		temp_numberOfPassengers = scenario[0][2]
		
		#Check scenario response, whether or not pedestrian are injuried
		scenarioDice = random.random()
		dead = 0

		probDeath=0
		if self.predAction==0:
			probDeath = scenario[0][1]
		else:
			probDeath = scenario[0][3]

		if (scenarioDice < probDeath):
			dead = 1
		
		if self.predAction==0:
			#utility = (temp_numberOfPassengers * selfish - temp_numberOfPedestrians * self.altruism   - temp_numberOfPedestrians * computeCost * conf.costPedestrian * dead )
			utility = (temp_numberOfPassengers * selfish + temp_numberOfPedestrians * (1 - dead) * self.altruism) - (temp_numberOfPassengers * selfish * (1 - scenario[0][3]) + temp_numberOfPedestrians * self.altruism ) - temp_numberOfPedestrians * computeCost * conf.costPedestrian * dead 
			#utility = (temp_numberOfPassengers * selfish + temp_numberOfPedestrians * (1 - dead) * self.altruism - dead * temp_numberOfPedestrians * (self.altruism + computeCost * conf.costPedestrian)) 
		else:
			#utility = (temp_numberOfPedestrians * self.altruism  - temp_numberOfPassengers * selfish)
			utility = (temp_numberOfPassengers * selfish * (1 - dead) + temp_numberOfPedestrians * self.altruism) - (temp_numberOfPassengers * selfish + temp_numberOfPedestrians * self.altruism * (1 - scenario[0][1]) - temp_numberOfPedestrians * computeCost * conf.costPedestrian * scenario[0][1] )
			#utility = ((1-dead)*temp_numberOfPassengers * selfish  - dead * temp_numberOfPassengers * selfish + temp_numberOfPedestrians * self.altruism )
		
		# This is used to normalize the fitness value into 0-1
		'''utility_straight_max_pass=((conf.numberOfPassengers+1) * selfish - 1 * self.altruism)
		utility_straight_min_pass=(1 * selfish - (conf.numberOfPedestrians+1) * self.altruism - (conf.numberOfPedestrians+1) * conf.costPedestrian - conf.STIGMA)
		utility_turn_max_pass = (conf.numberOfPedestrians+ 1 ) * self.altruism  - 1 * selfish + conf.HONOR
		utility_turn_min_pass = (1 * self.altruism - (conf.numberOfPassengers + 1) * selfish )'''

		'''utility_straight_max_pass=(temp_numberOfPassengers * selfish - temp_numberOfPedestrians * self.altruism)
		utility_straight_min_pass=(temp_numberOfPassengers * selfish - temp_numberOfPedestrians * self.altruism - temp_numberOfPedestrians * conf.costPedestrian)
		utility_turn_max_pass = temp_numberOfPedestrians * self.altruism  - temp_numberOfPassengers * selfish
		utility_turn_min_pass = (temp_numberOfPedestrians * self.altruism - temp_numberOfPassengers * selfish )'''

		utility_straight_max_pass = (temp_numberOfPassengers * selfish + temp_numberOfPedestrians * 1 * self.altruism) - (temp_numberOfPassengers * selfish * (1 - scenario[0][3]) + temp_numberOfPedestrians * self.altruism ) - temp_numberOfPedestrians * computeCost * conf.costPedestrian * 0 
		utility_straight_min_pass = (temp_numberOfPassengers * selfish + temp_numberOfPedestrians * 0 * self.altruism) - (temp_numberOfPassengers * selfish * (1 - scenario[0][3]) + temp_numberOfPedestrians * self.altruism ) - temp_numberOfPedestrians * 1 * conf.costPedestrian * 1
		utility_turn_max_pass = (temp_numberOfPassengers * selfish * 1 + temp_numberOfPedestrians * self.altruism) - (temp_numberOfPassengers * selfish + temp_numberOfPedestrians * self.altruism * (1 - scenario[0][1]) - temp_numberOfPedestrians * computeCost * conf.costPedestrian * scenario[0][1])
		utility_turn_min_pass = (temp_numberOfPassengers * selfish * 0 + temp_numberOfPedestrians * self.altruism) - (temp_numberOfPassengers * selfish + temp_numberOfPedestrians * self.altruism * (1 - scenario[0][1]) - temp_numberOfPedestrians * computeCost * conf.costPedestrian * scenario[0][1])

		'''utility_straight_max_pass = (temp_numberOfPassengers * selfish + temp_numberOfPedestrians *  self.altruism) 
		utility_straight_min_pass = (temp_numberOfPassengers * selfish + temp_numberOfPedestrians *  self.altruism - 2 * 1 * temp_numberOfPedestrians *  self.altruism - 1 * temp_numberOfPedestrians * computeCost * conf.costPedestrian)
		utility_turn_max_pass = (temp_numberOfPassengers * selfish - 0*temp_numberOfPassengers * selfish  - 0 * temp_numberOfPassengers * selfish + temp_numberOfPedestrians * self.altruism )
		utility_turn_min_pass = (temp_numberOfPassengers * selfish - 1*temp_numberOfPassengers * selfish  - 1 * temp_numberOfPassengers * selfish + temp_numberOfPedestrians * self.altruism )'''
		
		'''print(f"STRAIGHT MAX: {utility_straight_max_pass} \t STRAIGHT MIN: {utility_straight_min_pass} \t")
		print(f"TURN MAX: {utility_turn_max_pass} \t TURN MIN: {utility_turn_min_pass} \t")
		print(f"altruism: {temp_numberOfPassengers * selfish} \t penalty: {temp_numberOfPedestrians * self.altruism} \t cost {temp_numberOfPedestrians * conf.costPedestrian}")
		print(f"utility: {temp_numberOfPassengers * selfish} \t penalty: {- temp_numberOfPedestrians * self.altruism} \t")
		print(f"coswt: {- temp_numberOfPedestrians * conf.costPedestrian} \t stigme: {conf.STIGMA} \t")
		print(f"FITNESS: {utility} \t predAction: {self.predAction} \n")'''

		utility_max_value = max(utility_straight_max_pass,utility_turn_max_pass,utility_straight_min_pass,utility_turn_min_pass)
		
		utility_min_value = min(utility_straight_max_pass,utility_turn_max_pass,utility_straight_min_pass,utility_turn_min_pass)
		
		range_utility=(utility_max_value-utility_min_value)
		
		
		#self.fitness = (utility-utility_min_value)/range_utility + reward
		#self.fitness = utility 
		self.fitness = (utility - utility_min_value)/range_utility
					
		#print(f"Action: {predAction} \tknob: {self.knob:.4f} \tPed: {temp_numberOfPedestrians:.4f} \tPass: {temp_numberOfPassengers:.4f} \tdead: {dead} \treward: {reward} \tfitness: {self.fitness:.4f}")
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
	#m_model.add(Dense(5,  activation='relu'))
	#m_model.add(Dense(1,  activation='tanh'))
	m_model.add(Dense(1))

	# Compile Neural Network
	m_model.compile(optimizer='adam', loss='categorical_crossentropy')

	#m_model.summary()
	return m_model

def generate_first_population_randomly(conf):
	"""
	Creates an Initial Random Population
	:param population_size:
	:return:
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
					delta = random.uniform(-delta,delta)
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
	#for l in children.nn.layers:
	#print("Prima del crossover")
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
	#print("Dopo del crossover")	
	return children

def tournament_selection(population,selectionSize):
	
	list_index_a=random.sample(range(len(population)),selectionSize)
	list_index_b=random.sample(range(len(population)),selectionSize)
	
	'''print("Original")
	print([p.fitness for p in population])
	
	print("A")
	print([p.fitness for p in population_a])
	print("B")
	print([p.fitness for p in population_b])'''
	
	population_result=[]
	for i in range(selectionSize):
		index_a=list_index_a[i]
		index_b=list_index_b[i]
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
		parents = tournament_selection(population,conf.BEST_CANDIDATES_COUNT)
	
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
			
			#Number of pedestrian varies in 1 - numberOfPedestrians+1
			nPed=random.randint(1,conf.numberOfPedestrians+1) 
			#nPed = 1
			#Number of passengers varies in 1 - numberOfPedestrians+1
			nPass=random.randint(1,conf.numberOfPassengers+1)
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
			df=df.append(df_temp)
			
	
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
