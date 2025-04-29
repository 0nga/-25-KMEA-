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
import copy

import sys, getopt

import random
random.seed(42)


class Individual:

	"""
	Define an Individual
	Rappresenta un individuo nel tuo algoritmo genetico:
	Ha una rete neurale (self.nn) che produce il valore knob da un input di scenario.
	Il metodo computeFitness calcola quanto è "buona" la decisione presa (fitness) in base a:
	- il numero di passeggeri e pedoni,
	- le probabilità di morte,
	- il valore di altruismo.
	"""
	def __init__(self,conf):
		
		self.nn = make_nn_individual()
		self.fitness = 0
		self.age = 0
		self.altruism = conf.ALTRUISM
		self.knob = random.random()
		#print("Knob initialization",self.knob)

	
	def computeFitness(self, scenario, conf,avgKnobLevel=0):
		'''
		Funzione cuore del comportamento dell’agente:
		- Normalizza i dati di input.
		- La rete neurale produce un valore di knob.
		- Decide se andare dritto o svoltare a seconda del peso che knob dà a vite umane.
		- Calcola l’utilità dell’azione presa (basata su altruismo, perdite umane, costi).
		- Aggiunge una ricompensa sociale (honor/stigma) se l’agente si comporta diversamente dalla media della popolazione.
		'''
		# Standardization of the input for NN
		scenario_copia=copy.deepcopy(scenario)
		scenario_copia[0][0]=(scenario_copia[0][0]-3.5)/1.7
		scenario_copia[0][2]=(scenario_copia[0][2]-3.5)/1.7
		#print("Scenario pre predict ",scenario_copia)
		self.knob = self.nn.predict(scenario_copia)[0][0]
		#print("Predicction ",self.nn.predict(scenario_copia))
		predAction = 0;

		#Evaluate scenario
		evaluatePedestrian = scenario[0][0] * conf.probDeathPedestrians
		evaluatePassengers = scenario[0][2] * conf.probDeathPassengers
		
		#predAction=1 means turn, predAction=0 means go straight
		if (self.knob * evaluatePassengers < evaluatePedestrian * (1-self.knob)):
			predAction = 1
			
		selfish = 1 - self.altruism
		
		temp_numberOfPedestrians = scenario[0][0];
		temp_numberOfPassengers = scenario[0][2];
		
		#Check scenario response, whether or not pedestrian are injuried
		scenarioDice = random.random()
		dead = 0
		if (scenarioDice < scenario[0][1]):
			dead = 1
		
		# Check whether the cost should be computed or not
		# If the probability of harming pedestrian is greater than the probability of harming passenger than cost should 
		# be computed 
		computeCost=0
		if (scenario[0][1] > scenario[0][3] ):
			computeCost=1

			
		utility = (temp_numberOfPassengers * selfish - temp_numberOfPedestrians * self.altruism   - temp_numberOfPedestrians * computeCost * conf.costPedestrian * dead )*(1-predAction)
		utility += (temp_numberOfPedestrians * self.altruism  - temp_numberOfPassengers * selfish)*(predAction)
		
		'''utility_straight_max_pass=((conf.numberOfPassengers+1) * selfish - self.altruism   - conf.costPedestrian)
		utility_straight_min_pass = (-(conf.numberOfPedestrians+1) * self.altruism   - (conf.numberOfPedestrians+1) * conf.costPedestrian)
		
		utility_turn_max_pass = (self.altruism  - (conf.numberOfPassengers+1) * selfish)
		utility_turn_min_pass = ((conf.numberOfPedestrians+1) * self.altruism)
		
		utility_max_value = max(utility_straight_max_pass,utility_turn_max_pass,utility_straight_min_pass,utility_turn_min_pass)
		
		utility_min_value = min(utility_straight_max_pass,utility_turn_max_pass,utility_straight_min_pass,utility_turn_min_pass)
		
		range_utility=(utility_max_value-utility_min_value)'''
		
		'''
		Compute the action performed by an average individual
		'''
		avgAction=0
		if(avgKnobLevel * evaluatePassengers < evaluatePedestrian * (1-avgKnobLevel)):
			avgAction = 1
			
		reward=0
			
		'''
		 If on average community choose a selfish action and I choose an altruistic action then the reward is honorable and weighted with my altruistic behaviour 
		 ''' 
		if(avgAction == 0 and predAction == 1):
			reward = conf.HONOR #*altruism

		'''
		 If on average community choose an altruistic action and I choose a selfish action then the reward is a stigma weighted with my altruistic behaviour 
		 '''
		if(avgAction == 1 and predAction == 0):
			reward = conf.STIGMA #* selfish

		
		#self.fitness = (utility-utility_min_value)/range_utility + reward
		self.fitness = utility + reward
					
		#print(f"Action: {predAction} \tknob: {self.knob:.4f} \tPed: {temp_numberOfPedestrians:.4f} \tPass: {temp_numberOfPassengers:.4f} \tdead: {dead} \treward: {reward} \tfitness: {self.fitness:.4f}")
		return predAction


def make_nn_individual():
	m_model = Sequential()

	'''
	Crea una rete neurale molto semplice:
	- Input: 6 variabili (numero di pedoni, probabilità di morte pedoni, numero di passeggeri, probabilità di morte passeggeri, altruismo, knob).
	- Output: un singolo valore in [-1, 1] (attivazione tanh), usato come "knob".

	inp = Input(shape=(6,1))
	x = Dense(3, activation='relu', name='dense1')(inp)
	out = Dense(1, activation='tanh', name='dense2')(x)
	m_model=Model(inp, out)'''

	m_model.add(Dense(3, input_dim=6, activation='relu'))
	#m_model.add(Dense(5,  activation='relu'))
	m_model.add(Dense(1,  activation='tanh'))

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

        result.append(Individual(conf))

    run_stop = time.time()
    print(f"Done > Takes {run_stop-run_start} sec")

    return result

def mutate_chromosome(conf,individual=None):
	"""
	Randomnly mutate individual chromosome
	introduce mutazioni nei pesi della rete neurale.
	:param population: Current Population
	:return: A new population
	"""
	#Apply mutation to each weight of the NN 
	for l in individual.nn.layers:
		weights = l.get_weights()
		for i in range(len(weights[0])):
			for j in range(len(weights[0][i])):
				#Compute random value in the delta interval
				if random.random()>conf.HIDDEN_LAYER_MUTATION_PROBABILITY:
					delta = weights[0][i][j] * conf.HIDDEN_LAYER_MUTATION_RANGE
					delta = random.uniform(-delta,delta)
					weights[0][i][j] += delta
					
		l.set_weights(weights)
		
	return individual

def generate_children(conf,mother: Individual, father: Individual) -> Individual:
	"""
	Generate a New Children based Mother and Father Genomes
	combina i pesi di due genitori
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
	
	'''
	seleziona i migliori tramite un torneo.
	'''
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
	crea una nuova generazione selezionando, incrociando e mutando gli individui.
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
		
		new_population.append(
            mutate_chromosome(conf,
                generate_children(conf,
                    mother=parents[parent_a],
                    father=parents[parent_b]
                )
            )
        )	
	return new_population

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
	
def save_accuracy(df,conf,gen=0):
	
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
	
	df.to_csv(os.path.join(pathLog, "accuracy_"+s+".txt"), sep="\t", decimal=",")

def save_options(conf):
	try:
		os.makedirs(conf.path)
	except OSError:
		print ("Creation of the directory %s failed" % conf.path)
	else:
		print ("Successfully created the directory %s" % conf.path)
		
	f = open(os.path.join(conf.path, "out.txt"), "w")
	f.write("HIDDEN_LAYER_COUNT: " + str(conf.HIDDEN_LAYER_COUNT) + "\n")
	f.write("HIDDEN_LAYER_NEURONS: " + str(conf.HIDDEN_LAYER_NEURONS) + "\n")
	f.write("HIDDEN_LAYER_RATE: "+ str(conf.HIDDEN_LAYER_RATE) + "\n")
	f.write("HIDDEN_LAYER_ACTIVATIONS: "+ str(conf.HIDDEN_LAYER_ACTIVATIONS) + "\n")
	f.write("HIDDEN_LAYER_TYPE: "+ str(conf.HIDDEN_LAYER_TYPE) + "\n")
	f.write("MODEL_OPTIMIZER: "+ str(conf.MODEL_OPTIMIZER) + "\n")

	f.write("MAX_GENERATIONS: "+ str(conf.MAX_GENERATIONS) + "\n")
	f.write("POPULATION_SIZE: "+ str(conf.POPULATION_SIZE) + "\n")
	f.write("BEST_CANDIDATES_COUNT: "+ str(conf.BEST_CANDIDATES_COUNT) + "\n")
	f.write("RANDOM_CANDIDATES_COUNT: "+ str(conf.RANDOM_CANDIDATES_COUNT) + "\n")
	f.write("OPTIMIZER_MUTATION_PROBABILITY: "+ str(conf.OPTIMIZER_MUTATION_PROBABILITY) + "\n")
	f.write("HIDDEN_LAYER_MUTATION_PROBABILITY: " + str(conf.HIDDEN_LAYER_MUTATION_PROBABILITY) + "\n")
	f.write("HIDDEN_LAYER_MUTATION_RANGE: "+ str(conf.HIDDEN_LAYER_MUTATION_RANGE) + "\n")

	f.write("costPedestrian: "+ str(conf.costPedestrian) + "\n")
	f.write("costPassengers: "+ str(conf.costPassengers) + "\n")
	f.write("altruisticBehavior: "+ str(conf.ALTRUISM) + "\n")
	f.write("numberOfPedestrians: "+ str(conf.numberOfPedestrians+1) + "\n")
	f.write("numberOfPassengers: "+ str(conf.numberOfPassengers+1) + "\n")
	f.write("probDeathPedestrians: "+ str(conf.probDeathPedestrians) + "\n")
	f.write("probDeathPassengers: " + str(conf.probDeathPassengers) + "\n")
	f.write("STIGMA: " + str(conf.STIGMA) + "\n")
	f.write("HONOR: "+ str(conf.HONOR) + "\n")
	f.close()
	


	'''
	Nel main():
	- Vengono lette opzioni da terminale (es. numero generazioni, altruismo).
	- Viene salvata la configurazione.
	- Parte l'evoluzione per MAX_GENERATIONS, salvando i risultati in CSV.
	- Ogni individuo viene testato in scenari randomici e valutato in base alla scelta morale.
	'''
def main():

	conf=Configuration()
	opts, args = getopt.getopt(sys.argv[1:],"ha:o:g:p:e:r")
	print(f"Arguments count: {len(sys.argv)}")
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("'test.py -g <numGenerations> -p <numIndividuals> -e <percElite> -r <percRandom>'")
			sys.exit()
		elif opt in ("-a",):	
			conf.set_altruism(float(arg))
			print(f"ALTRUISM: {conf.ALTRUISM}")
			#sys.exit()
		elif opt in ("-g", "--gen"):
			conf.MAX_GENERATIONS=int(arg)
			print(f"MAX_GENERATIONS: {conf.MAX_GENERATIONS}")
		elif opt in ("-p",):
			conf.set_population_size(int(arg))
			print(f"POPULATION_SIZE: {conf.POPULATION_SIZE}")
		elif opt in ("-o",):	
			conf.path = arg
			print(f"path: {conf.path}")
		elif opt in ("-e",):	
			conf.probDeathPedestrians = float(arg)
			print(f"probPed: {conf.probDeathPedestrians}")
		elif opt in ("-r",):	
			conf.randomizeAltruism = True
			print(f"randomizeAltruism: {conf.randomizeAltruism}")
			
	#sys.exit()		
	save_options(conf)
	
	# >>>>>> Genetic Algorithm Section <<<<<<
	print("\n********** Genetic Algorithm **********")
	population_new = generate_first_population_randomly(conf)

	# Run Each Generation
	for current_generation in range(conf.MAX_GENERATIONS):
		population = population_new
		print(f"[+] Generation {current_generation+1} of {conf.MAX_GENERATIONS}")
		i = 0
		
		'''df=pd.DataFrame(columns=["AltruismLevel",
					"KnobLevel",
					"numberOfPedestrians",	
					"numberOfPassengers",
					"Fitness",
					"Age", 
					"probPed", 
					"probPass"])'''
		
		df=pd.DataFrame(columns=["numberOfPedestrians",
					"probPed", 
					"numberOfPassengers",
					"probPass",
					"AltruismLevel",
					"KnobLevel",
					"KnobLevel_pred",
					"Fitness",
					'convieneSvolta',
					"predAction"
					])
		
		# >>>>>> Evaluation Phase <<<<<<
		print(f"\tEvaluating Population: ", end='', flush=True)
		evaluation_start = time.time()
		
		for p in population:
			
			#Number of pedestrian varies in 1 - numberOfPedestrians+1
			#nPed=random.randint(1,conf.numberOfPedestrians+1) 
			nPed = 1
			#Number of passengers varies in 1 - numberOfPedestrians+1
			#nPass=random.randint(1,conf.numberOfPassengers+1)
			nPass = 1

			probDeathPedestrians = random.random()
			probDeathPassengers = random.random()

			scenario = [nPed,probDeathPedestrians,nPass,probDeathPassengers,p.altruism, p.knob]
			#print("Prima",scenario)
			scenario = np.array(scenario)
			scenario = scenario.reshape((1,6))
			predAction = p.computeFitness(scenario,conf)
			
			scenario = np.append(scenario,p.knob)
			scenario = np.append(scenario,p.fitness)
			#print("Dopo",scenario)
			#l.numberOfPedestrians*l.AltruismLevel - l.numberOfPassengers * (1-l.AltruismLevel)
			u_svolta= scenario[0] * scenario[4] - scenario[2] * (1 - scenario[4])
			#u_dritto=l.numberOfPassengers * (1-l.AltruismLevel) - l.numberOfPedestrians*l.AltruismLevel - l.numberOfPedestrians
			computeCost=0
			if (scenario[1] > scenario[3] ):
				computeCost=1
			u_dritto= scenario[2] * (1-scenario[4]) - scenario[0]*scenario[4] - scenario[0] * computeCost * conf.costPedestrian

			#predAction=1 means turn, predAction=0 means go straight
			'''convieneSvolta = 0
			if (u_svolta > u_dritto):
				convieneSvolta = 1 '''

			convieneSvolta = (u_svolta > u_dritto)
			#print("Conviene svoltare:",convieneSvolta)
			scenario=np.append(scenario,convieneSvolta)
			scenario=np.append(scenario,predAction)
			#isSvolta =(l.KnobLevel*l.numberOfPassengers < (1-l.KnobLevel)*prob*l.numberOfPedestrians)
			#isSvolta =(scenario[0][5] * scenario[0][2] < (1-scenario[0][5]) * prob * scenario[0][0])
			#print(scenario)
			df_temp=pd.DataFrame([scenario],columns=["numberOfPedestrians",
					"probPed", 
					"numberOfPassengers",
					"probPass",
					"AltruismLevel",
					"KnobLevel",
					"KnobLevel_pred",
					"Fitness",
					'convieneSvolta',
					"predAction"
					])
			##############################DEPRECATO#########################################

			#df = df.append(df_temp)
			df = pd.concat([df, df_temp], ignore_index=True)
			
			#p = mutate_chromosome(conf,p)
			
		evaluation_stop = time.time()
		print(f"Done > Takes {evaluation_stop - evaluation_start} sec")
		
		# Compute Generation Metrics
		gen_score_avg = np.average([item.fitness for item in population])
		gen_score_min = np.min([item.fitness for item in population])
		gen_score_max = np.max([item.fitness for item in population])
		gen_score_knob = np.average([item.knob for item in population])

		print(f"\tWorst Score:{gen_score_min:.4f} | Average Score:{gen_score_avg:.4f} | Best Score:{gen_score_max:.4f} | Knob Score:{gen_score_knob:.4f}")
		
		save_results(df,conf,current_generation)
		# >>>>>> Genetic Selection, Children Creation and Mutation <<<<<<
		population_new = evolve_population(population,conf)

	'''print("All scenarios for last generation:")
	print(df)
	print("All scenarios for last generation:")'''
	df_temp=df.drop(["KnobLevel_pred","Fitness","convieneSvolta","predAction"],axis=1)
	df_temp.numberOfPedestrians=(df_temp.numberOfPedestrians-3.5)/1.7
	df_temp.numberOfPassengers=(df_temp.numberOfPassengers-3.5)/1.7
	#print(df_temp)
	accuracyList=pd.DataFrame(columns=['individual','fp','fn','tp','tn'])
	accuracyList_pred=pd.DataFrame(columns=['individual','true_y','pred_y'])
	i=0
	for p in population:
		knob_prediction=p.nn.predict(df_temp)
		knob_prediction = knob_prediction.reshape(len(knob_prediction))
		'''print("Knob", knob_prediction)
		print("Knob", df.probPass)
		print("Knob", df.numberOfPassengers)
		print("VAlore sterza", knob_prediction * df.probPass * df.numberOfPassengers )
		print("VAlore dritto",(1-knob_prediction) * df.probPed * df.numberOfPedestrians)'''

		isSvolta = knob_prediction[0] * df.probPass * df.numberOfPassengers < (1-knob_prediction) * df.probPed * df.numberOfPedestrians
		#accuracy = (df.convieneSvolta == isSvolta)

		'''print("Conviene: " , df.convieneSvolta)
		print("Azionbe: ", isSvolta)'''
		tp = sum((df.convieneSvolta==True) & (isSvolta==True))
		tn = sum((df.convieneSvolta==False) & (isSvolta==False))
		fp = sum((df.convieneSvolta==False) & (isSvolta==True))
		fn = sum((df.convieneSvolta==True) & (isSvolta==False))
		
		#print(f"TP: {tp}\t TN: {tn}\t FP: {fp}\t FN: {fn}")
		### DEPRECATA ###
		# accuracyList=accuracyList.append({'individual':p,'fp':fp,'fn':fn,'tp':tp,'tn':tn},ignore_index=True)
		accuracyList = pd.concat([accuracyList, pd.DataFrame([{'individual': p, 'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn}])], ignore_index=True)

		i = i + 1
		
	#print(accuracyList)
	save_accuracy(accuracyList,conf,conf.MAX_GENERATIONS)
		

	'''
	Nel dataframe df salvi per ogni individuo:
	- Caratteristiche dello scenario.
	- Livello di altruismo.
	- Valore di knob previsto.
	- Fitness.
	- Se "conviene" svoltare logicamente.
	- Cosa ha fatto realmente (azione predetta dalla NN).
	'''
	
if __name__ == "__main__":
	
    # Disable Tensorflow Warning Messages
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	
	#opts, args = getopt.getopt(sys.argv[1:],"hg:p:e:r:a:o:")
	
	# Run Program
	print(f"\tStart simulation: ", end='', flush=True)
	simulation_start = time.time()
	main()
	simulation_stop = time.time()
	print(f"Done > Takes {simulation_stop - simulation_start} sec")