import os
import random
import pandas as pd
import numpy as np
import time
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, InputLayer, Input
from keras import backend as K
from Configuration import Configuration
import shutil

from Individual import *

import sys, getopt

import random
random.seed(42)
	
def main():

	# Creo l'oggetto che conterrà la configurazione
	conf = Configuration() 

	# Gestione dei flag passati nel comando
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
			conf.set_path(arg)
			print(f"path: {conf.path}")
		elif opt in ("-e",):	
			conf.probDeathPedestrians = float(arg)
			print(f"probPed: {conf.probDeathPedestrians}")
		elif opt in ("-r",):	
			conf.randomizeAltruism = True
			print(f"randomizeAltruism: {conf.randomizeAltruism}")
		
	# Salvataggio opzioni
	save_options(conf)
	
	# >>>>>> Genetic Algorithm Section <<<<<<
	print("\n********** Genetic Algorithm **********")

	# --- DELETE AND RECREATE OUTPUT DIRECTORY AT THE START ---
	pathLog = os.path.join(conf.path, "logs", "0")
	try:
		if os.path.exists(pathLog):
			shutil.rmtree(pathLog)
		os.makedirs(pathLog, exist_ok=False)
		print(f"Output directory {pathLog} cleaned and recreated.")
	except OSError as e:
		print(f"Error cleaning/creating output directory {pathLog}: {e}")
		sys.exit(1) # Exit if there's an issue with the output directory

	# Funzione presente in individual.py
	population_new = generate_first_population_randomly(conf)

	# Run Each Generation
	for current_generation in range(conf.MAX_GENERATIONS):
		population = population_new
		print(f"[+] Generation {current_generation+1} of {conf.MAX_GENERATIONS}")
		i = 0
		
		df = pd.DataFrame(columns=["numberOfPedestrians",
					"probPed", 
					"numberOfPassengers",
					"probPass",
					"AltruismLevel",
					"KnobLevel",
					"Fitness",
					'convieneSvolta',
					"predAction"
					])
		df_scenari = createScenarios(conf,population,conf.randomizeAltruism)

		scenari, scaler = standardize(df_scenari)

		# >>>>>> Evaluation Phase <<<<<<
		print(f"\tEvaluating Population: ", end='', flush=True)
		evaluation_start = time.time()

		'''
		UPDATE FITNESS
		'''
		predAction_list = []
		ind = 0

		for p in population:
			scenario = np.array(scenari[ind])
			scenario = scenario.reshape((1,5))
			predAction = p.computeFitness(scenario, conf, scaler)
			predAction_list.append(predAction)
			ind += 1
			
		evaluation_stop = time.time()
		print(f"Done > Takes {evaluation_stop - evaluation_start} sec")
		
		gen_score_knob = np.average([item.knob for item in population])

		'''
		COMPUTE SELF-ESTEEM
		'''
		ind = 0
		for p in population:
			p.computeSelfEsteem(conf,gen_score_knob)
			
			scenario = np.array(df_scenari.iloc[ind])
			scenario = np.append(scenario,p.knob)
			scenario = np.append(scenario,p.fitness)

			computeCost=0
			if (scenario[1] * scenario[0] > scenario[3] * scenario[2]):
				computeCost=1

			u_svolta = scenario[0] * scenario[4] * scenario[1] - scenario[2] * (1 - scenario[4]) + scenario[0] * computeCost * conf.costPedestrian * scenario[1]
			u_dritto = scenario[2] * (1-scenario[4]) * scenario[3] - scenario[0]*scenario[4] - scenario[0] * computeCost * conf.costPedestrian

			convieneSvolta = (u_svolta > u_dritto)
			
			scenario=np.append(scenario,convieneSvolta)
			scenario=np.append(scenario,predAction_list[ind])
			
			df_temp=pd.DataFrame([scenario],columns=["numberOfPedestrians",
					"probPed", 
					"numberOfPassengers",
					"probPass",
					"AltruismLevel",
					"KnobLevel",
					"Fitness",
					'convieneSvolta',
					"predAction"
					])

			df = pd.concat([df, df_temp], ignore_index=True)
			ind += 1

		# Compute Generation Metrics
		gen_score_avg = np.average([item.fitness for item in population])
		gen_score_min = np.min([item.fitness for item in population])
		gen_score_max = np.max([item.fitness for item in population])
		
		print(f"\tWorst Score:{gen_score_min:.4f} | Average Score:{gen_score_avg:.4f} | Best Score:{gen_score_max:.4f} | Knob Score:{gen_score_knob:.4f}")
		
		save_results(df,conf,current_generation)
		# >>>>>> Genetic Selection, Children Creation and Mutation <<<<<<
		population_new = evolve_population(population,conf)

	columns_to_drop = ["Unnamed: 0", "KnobLevel", "Fitness", "convieneSvolta", "predAction"]
	df_temp = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
	df_temp, _ = standardize(df_temp) 

	accuracyList = pd.DataFrame(columns=['individual','fp','fn','tp','tn'])
	accuracyList_pred = pd.DataFrame(columns=['true_y','pred_y'])
	i = 0

	for p in population:
		#Compute Prediction
		knob_prediction = p.nn.predict(df_temp)
		knob_prediction = knob_prediction.reshape(len(knob_prediction))
	

		isSvolta = knob_prediction * df.probPass * df.numberOfPassengers < (1-knob_prediction) * df.probPed * df.numberOfPedestrians
		'''
		isSvolta = knob_prediction * df.probPass * df.numberOfPassengers < (1-knob_prediction) * df.probPed * df.numberOfPedestrians
		'''
	
		tp = sum((df.convieneSvolta==True) & (isSvolta==True))
		tn = sum((df.convieneSvolta==False) & (isSvolta==False))
		fp = sum((df.convieneSvolta==False) & (isSvolta==True))
		fn = sum((df.convieneSvolta==True) & (isSvolta==False))
		
		accuracyList = pd.concat([accuracyList, pd.DataFrame([{'individual':p,'fp':fp,'fn':fn,'tp':tp,'tn':tn}])], ignore_index=True)
		
		temp = pd.DataFrame(list(zip(df.convieneSvolta.values,isSvolta.values)),columns=['true_y','pred_y'])
		accuracyList_pred = pd.concat([accuracyList_pred, temp], ignore_index=True)
		i = i + 1
		
	save_accuracy(accuracyList,conf,None,conf.MAX_GENERATIONS)
	save_accuracy(accuracyList_pred,conf,"detailed_acccuracy.txt")
		
	
if __name__ == "__main__":
	
	# Run Program
	print(f"\tStart simulation: ", end='', flush=True)
	simulation_start = time.time()
	main()
	simulation_stop = time.time()
	print(f"Done > Takes {simulation_stop - simulation_start} sec")