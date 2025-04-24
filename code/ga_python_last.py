import os
import random
#import gym
import pandas as pd
import numpy as np
#import tensorflow as tf
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras import backend as K

import sys, getopt

import random
random.seed(42)

path="/Users/aloreggia/Downloads/test/pythonTest"
path="/Users/aloreggia/Downloads/test/pythonTest_altruism_1/"

# Define Hyperparameters for NN
#HIDDEN_LAYER_COUNT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
HIDDEN_LAYER_COUNT = [1]
#HIDDEN_LAYER_NEURONS = [8, 16, 24, 32, 64, 128, 256, 512]
HIDDEN_LAYER_NEURONS = [6]
#HIDDEN_LAYER_RATE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HIDDEN_LAYER_RATE = [0.1]
#HIDDEN_LAYER_ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'linear', 'softmax']
HIDDEN_LAYER_ACTIVATIONS = ['relu']
#HIDDEN_LAYER_TYPE = ['dense', 'dropout']
HIDDEN_LAYER_TYPE = ['dense']
MODEL_OPTIMIZER = ['adam', 'rmsprop']

# Define Genetic Algorithm Parameters
MAX_GENERATIONS = 100  # Max Number of Generations to Apply the Genetic Algorithm
POPULATION_SIZE = 100  # Max Number of Individuals in Each Population
BEST_CANDIDATES_COUNT = int(POPULATION_SIZE * 0.4)  # Number of Best Candidates to Use
RANDOM_CANDIDATES_COUNT = int(POPULATION_SIZE * 0.1)  # Number of Random Candidates (From Entire Population of Generation) to Next Population
OPTIMIZER_MUTATION_PROBABILITY = 0.1  # 10% of Probability to Apply Mutation on Optimizer Parameter
HIDDEN_LAYER_MUTATION_PROBABILITY = 0.1  # 10% of Probability to Apply Mutation on Hidden Layer Quantity
HIDDEN_LAYER_MUTATION_RANGE = 0.01  # Apply Mutation of 1%

costPedestrian = 1.0		# cost of pedestrian c_s
costPassengers = 1.0 		# cost of passengers c_t
ALTRUISM = 0.1			# altruistic behavior
numberOfPedestrians=5		# numberOfPedons
numberOfPassengers=5		# numberOfPassengers
probDeathPedestrians=1.0	# probDeathPedestrians
probDeathPassengers=1.0		# probDeathPassengers
STIGMA = -1.0
HONOR = 1.0

class Individual:

	"""
	Define an Individual
	"""
	def __init__(self):
		self.nn = make_nn_individual()
		self.fitness = 0
		self.age = 0
		self.altruism = ALTRUISM
		self.knob = random.random()

	def computeFitness(self, scenario, avgKnobLevel=0):
		self.knob=self.nn.predict(scenario)[0][0]
		predAction = 0;

		#Evaluate scenario
		evaluatePedestrian = scenario[0][0] * probDeathPedestrians
		evaluatePassengers = scenario[0][2] * probDeathPassengers
		
		if (self.knob * evaluatePassengers < evaluatePedestrian * (1-self.knob)):
			predAction = 1
			
		selfish = 1 - self.altruism
		
		temp_numberOfPedestrians = scenario[0][0];
		temp_numberOfPassengers = scenario[0][2];
		
		scenarioDice = random.random()
		dead = scenarioDice < scenario[0][1]
			
		utility = (temp_numberOfPassengers * selfish - temp_numberOfPedestrians * self.altruism   - temp_numberOfPedestrians* costPedestrian * dead )*(1-predAction)
		utility += (temp_numberOfPedestrians * self.altruism  - temp_numberOfPassengers * selfish)*(predAction);
		
		avgAction=0;
		if(avgKnobLevel * evaluatePassengers < evaluatePedestrian * (1-avgKnobLevel)):
			avgAction = 1;
			
		reward=0;
			
		'''
		 If on average community choose a selfish action and I choose an altruistic action then the reward is honorable and weighted with my altruistic behaviour 
		 ''' 
		if(avgAction == 0 and predAction == 1):
			reward = HONOR *altruism

		'''
		 If on average community choose an altruistic action and I choose a selfish action then the reward is a stigma weighted with my altruistic behaviour 
		 '''
		if(avgAction == 1 and predAction == 0):
			reward = STIGMA * selfish

		
		self.fitness = utility
					
		#print(f"Action: {predAction} \tknob: {self.knob:.4f} \tPed: {temp_numberOfPedestrians:.4f} \tPass: {temp_numberOfPassengers:.4f} \tdead: {dead} \treward: {reward} \tfitness: {self.fitness:.4f}")

def create_random_layer():
    """
    Creates a new Randomly Generated Layer
    :return:
    """

    layer_layout = LayerLayout(
        layer_type=HIDDEN_LAYER_TYPE[random.randint(0, len(HIDDEN_LAYER_TYPE) - 1)]
    )

    if layer_layout.layer_type == 'dense':
        layer_layout.neurons = HIDDEN_LAYER_NEURONS[random.randint(0, len(HIDDEN_LAYER_NEURONS) - 1)]
        layer_layout.activation = HIDDEN_LAYER_ACTIVATIONS[random.randint(0, len(HIDDEN_LAYER_ACTIVATIONS) - 1)]

    elif layer_layout.layer_type == 'dropout':
        layer_layout.rate = HIDDEN_LAYER_RATE[random.randint(0, len(HIDDEN_LAYER_RATE) - 1)]

    return layer_layout

def make_nn_individual():
	m_model = Sequential()
	m_model.add(Dense(5, input_dim=6, activation='relu'))
	#m_model.add(Dense(3,  activation='relu'))
	m_model.add(Dense(1,  activation='sigmoid'))

	# Compile Neural Network
	m_model.compile(optimizer='adam', loss='categorical_crossentropy')

	#m_model.summary()
	return m_model

def generate_first_population_randomly(population_size=10):
    """
    Creates an Initial Random Population
    :param population_size:
    :return:
    """

    print("[+] Creating Initial NN Model Population Randomly: ", end='')

    result = []
    run_start = time.time()

    for current in range(population_size):

        result.append(Individual())

    run_stop = time.time()
    print(f"Done > Takes {run_stop-run_start} sec")

    return result

def mutate_chromosome(individual=None):
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
				if random.random()>HIDDEN_LAYER_MUTATION_PROBABILITY:
					delta = weights[0][i][j] * HIDDEN_LAYER_MUTATION_RANGE
					delta = random.uniform(-delta,delta)
					weights[0][i][j] += delta
					
		l.set_weights(weights)
		
	return individual

def generate_children(mother: Individual, father: Individual) -> Individual:
	"""
	Generate a New Children based Mother and Father Genomes
	:param mother: Mother Individual
	:param father: Father Individual
	:return: A new Children
	"""
	
	children = Individual()
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
	
				
def evolve_population(population):
	"""
	Evolve and Create the Next Generation of Individuals
	:param population: Current Population
	:return: A new population
	"""
	#print(f"POPULATION_SIZE: {POPULATION_SIZE}")
	# Select N Best Candidates + Y Random Candidates. Kill the Rest of Chromosomes
	parents = []
	parents.extend(population[0:BEST_CANDIDATES_COUNT])  # N Best Candidates
	for rn in range(RANDOM_CANDIDATES_COUNT):
		parents.append(population[random.randint(0, POPULATION_SIZE - 1)])  # Y Random Candidate
		
	# Create New Population Through Crossover
	new_population = []
	new_population.extend(parents)

	while len(new_population) < POPULATION_SIZE:
		parent_a = random.randint(0, len(parents) - 1)
		parent_b = random.randint(0, len(parents) - 1)
		while parent_a == parent_b:
			parent_b = random.randint(0, len(parents) - 1)
		
		new_population.append(
            mutate_chromosome(
                generate_children(
                    mother=parents[parent_a],
                    father=parents[parent_b]
                )
            )
        )	
	return new_population

def save_results(df,gen=0,path=path):
	
	#print("Salva risultati in: "+path)
	s="%03d"%gen
	pathLog = path + "/logs/0"
	
	try:
		os.makedirs(pathLog)
	except OSError:
		#print ("Creation of the directory %s failed" % pathLog)
		print()
	else:
		#print ("Successfully created the directory %s" % pathLog)
		print()
		
	df.to_csv(os.path.join(pathLog, "gen_"+s+".txt"), sep="\t", decimal=",")
	
def save_options(path=path):
	
	try:
		os.makedirs(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s" % path)
	
	f = open(os.path.join(path, "out.txt"), "w")
	f.write("HIDDEN_LAYER_COUNT: " + str(HIDDEN_LAYER_COUNT) + "\n")
	f.write("HIDDEN_LAYER_NEURONS: " + str(HIDDEN_LAYER_NEURONS) + "\n")
	f.write("HIDDEN_LAYER_RATE: "+ str(HIDDEN_LAYER_RATE) + "\n")
	f.write("HIDDEN_LAYER_ACTIVATIONS: "+ str(HIDDEN_LAYER_ACTIVATIONS) + "\n")
	f.write("HIDDEN_LAYER_TYPE: "+ str(HIDDEN_LAYER_TYPE) + "\n")
	f.write("MODEL_OPTIMIZER: "+ str(MODEL_OPTIMIZER) + "\n")

	f.write("MAX_GENERATIONS: "+ str(MAX_GENERATIONS) + "\n")
	f.write("POPULATION_SIZE: "+ str(POPULATION_SIZE) + "\n")
	f.write("BEST_CANDIDATES_COUNT: "+ str(BEST_CANDIDATES_COUNT) + "\n")
	f.write("RANDOM_CANDIDATES_COUNT: "+ str(RANDOM_CANDIDATES_COUNT) + "\n")
	f.write("OPTIMIZER_MUTATION_PROBABILITY: "+ str(OPTIMIZER_MUTATION_PROBABILITY) + "\n")
	f.write("HIDDEN_LAYER_MUTATION_PROBABILITY: " + str(HIDDEN_LAYER_MUTATION_PROBABILITY) + "\n")
	f.write("HIDDEN_LAYER_MUTATION_RANGE: "+ str(HIDDEN_LAYER_MUTATION_RANGE) + "\n")

	f.write("costPedestrian: "+ str(costPedestrian) + "\n")
	f.write("costPassengers: "+ str(costPassengers) + "\n")
	f.write("altruisticBehavior: "+ str(ALTRUISM) + "\n")
	f.write("numberOfPedestrians: "+ str(numberOfPedestrians) + "\n")
	f.write("numberOfPassengers: "+ str(numberOfPassengers) + "\n")
	f.write("probDeathPedestrians: "+ str(probDeathPedestrians) + "\n")
	f.write("probDeathPassengers: " + str(probDeathPassengers) + "\n")
	f.write("STIGMA: " + str(STIGMA) + "\n")
	f.write("HONOR: "+ str(HONOR) + "\n")
	f.close()
	
def main():

	save_options(path)
	
	# >>>>>> Genetic Algorithm Section <<<<<<
	print("\n********** Genetic Algorithm **********")
	population = generate_first_population_randomly(population_size=POPULATION_SIZE)

	# Run Each Generation
	for current_generation in range(MAX_GENERATIONS):
		print(f"[+] Generation {current_generation+1} of {MAX_GENERATIONS}")
		i = 0
		
		df=pd.DataFrame(columns=["AltruismLevel","KnobLevel","numberOfPedestrians",	"numberOfPassengers","Fitness","Age"])
		
		# >>>>>> Evaluation Phase <<<<<<
		print(f"\tEvaluating Population: ", end='', flush=True)
		evaluation_start = time.time()
		#print(f"Individual in total: {len(population)}")
		for p in population:
			
			nPed=random.randint(1,numberOfPedestrians+1) 
			nPass=random.randint(1,numberOfPedestrians+1)
			scenario=[nPed,probDeathPedestrians,nPass,probDeathPassengers,p.altruism, p.knob]
			scenario=np.array(scenario)
			scenario=scenario.reshape((1,6))
			p.computeFitness(scenario)
			
			df=df.append(pd.DataFrame([[p.altruism,p.knob,nPed,nPass,p.fitness, p.age]],columns=["AltruismLevel","KnobLevel","numberOfPedestrians",	"numberOfPassengers","Fitness","Age"]))
			
			p = mutate_chromosome(p)
			
		
		#print(f"Done > Takes {evaluation_stop - evaluation_start} sec")
		
		# Sort Candidates by fitness
		population.sort(key=lambda x: x.fitness, reverse=True)
		
		# Compute Generation Metrics
		gen_score_avg = np.average([item.fitness for item in population])
		gen_score_min = np.min([item.fitness for item in population])
		gen_score_max = np.max([item.fitness for item in population])
		gen_score_knob = np.average([item.knob for item in population])

		print(f"\tWorst Score:{gen_score_min:.4f} | Average Score:{gen_score_avg:.4f} | Best Score:{gen_score_max:.4f} | Knob Score:{gen_score_knob:.4f}")
		
		save_results(df,current_generation,path)
		
		# >>>>>> Genetic Selection, Children Creation and Mutation <<<<<<
		#print(f"Individual in total prima: {len(population)}")
		population = evolve_population(population)
		#print(f"Individual in total dopo: {len(population)}")
		evaluation_stop = time.time()
		print(f"Done > Takes {evaluation_stop - evaluation_start} sec")
		
	
if __name__ == "__main__":

    # Disable Tensorflow Warning Messages
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	opts, args = getopt.getopt(sys.argv[1:],"hg:p:e:r:a:o:")
	print(f"Arguments count: {len(sys.argv)}")
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("'test.py -g <numGenerations> -p <numIndividuals> -e <percElite> -r <percRandom>'")
			sys.exit()
		elif opt in ("-g", "--gen"):
			MAX_GENERATIONS=eval(arg)
			print(f"MAX_GENERATIONS: {MAX_GENERATIONS}")
		elif opt in ("-p",):
			POPULATION_SIZE=eval(arg)
			BEST_CANDIDATES_COUNT = int(POPULATION_SIZE * 0.4)
			RANDOM_CANDIDATES_COUNT = int(POPULATION_SIZE * 0.1)
			print(f"POPULATION_SIZE: {POPULATION_SIZE}")
		elif opt in ("-e",):	
			BEST_CANDIDATES_COUNT = int(POPULATION_SIZE * eval(arg))
			print(f"BEST_CANDIDATES_COUNT: {BEST_CANDIDATES_COUNT}")
		elif opt in ("-r",):	
			RANDOM_CANDIDATES_COUNT = int(POPULATION_SIZE * eval(arg))
			print(f"RANDOM_CANDIDATES_COUNT: {RANDOM_CANDIDATES_COUNT}")
		elif opt in ("-a",):	
			ALTRUISM = eval(arg)
			print(f"ALTRUISM: {ALTRUISM}")
		elif opt in ("-o",):	
			path = arg
			print(f"path: {path}")
			
	#exit()
	
    # Run Program
	main()