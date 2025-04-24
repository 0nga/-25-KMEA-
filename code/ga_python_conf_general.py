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

from Individual import *

import sys, getopt

import random
random.seed(42)

	
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
			conf.set_path(arg)
			print(f"path: {conf.path}")
		elif opt in ("-e",):	
			conf.probDeathPedestrians = float(arg)
			print(f"probPed: {conf.probDeathPedestrians}")
		elif opt in ("-r",):	
			conf.randomizeAltruism = True
			print(f"randomizeAltruism: {conf.randomizeAltruism}")
		
	save_options(conf)
	#sys.exit()	
	
	# >>>>>> Genetic Algorithm Section <<<<<<
	print("\n********** Genetic Algorithm **********")
	population_new = generate_first_population_randomly(conf)

	# Run Each Generation
	for current_generation in range(conf.MAX_GENERATIONS):
		population = population_new
		print(f"[+] Generation {current_generation+1} of {conf.MAX_GENERATIONS}")
		i = 0
		
		df=pd.DataFrame(columns=["numberOfPedestrians",
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
		#print(df_scenari)
		scenari, scaler = standardize(df_scenari)
		#print(scenari)
		# >>>>>> Evaluation Phase <<<<<<
		print(f"\tEvaluating Population: ", end='', flush=True)
		evaluation_start = time.time()
		

		'''
		UPDATE FITNESS
		'''
		predAction_list=[]
		ind=0
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
			#print("Conviene svoltare:",convieneSvolta)
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
			
			##############################DEPRECATO#########################################

			#df = df.append(df_temp)
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

	'''print("All scenarios for last generation:")
	print(df)
	print("All scenarios for last generation:")'''
	
	df=pd.read_csv("/Users/aloreggia/Downloads/test/500ge/test_all.txt",decimal=",",delimiter="\t")
	df_temp=df.drop(["Unnamed: 0","KnobLevel","Fitness","convieneSvolta","predAction"],axis=1)
	#print(df_temp)
	df_temp, _ = standardize(df_temp) 
	#print(df_temp)
	accuracyList=pd.DataFrame(columns=['individual','fp','fn','tp','tn'])
	accuracyList_pred=pd.DataFrame(columns=['true_y','pred_y'])
	i=0
	for p in population:
		#Compute Prediction
		#print("Individual: " + str(p))
		#print(p)
		knob_prediction = p.nn.predict(df_temp)
		knob_prediction = knob_prediction.reshape(len(knob_prediction))
		'''print("Prediction")
		print(knob_prediction)
		print("scenari")
		print(df)
		print("senza [0]")
		print(1-knob_prediction)
		print("con [0]")
		print(1-knob_prediction[0])
		print("Knob", knob_prediction)
		print("probPass", df.probPass)
		print("numberOfPassengers", df.numberOfPassengers)
		print("VAlore sterza", knob_prediction * df.probPass * df.numberOfPassengers )
		print("VAlore dritto",(1-knob_prediction) * df.probPed * df.numberOfPedestrians)'''

		isSvolta = knob_prediction * df.probPass * df.numberOfPassengers < (1-knob_prediction) * df.probPed * df.numberOfPedestrians
		'''print("con [0]")
		print(isSvolta)
		print("senza [0]")
		isSvolta = knob_prediction * df.probPass * df.numberOfPassengers < (1-knob_prediction) * df.probPed * df.numberOfPedestrians
		print(isSvolta)'''
		#accuracy = (df.convieneSvolta == isSvolta)

		#print("Conviene: " , df.convieneSvolta)
		#print("Azionbe: ", isSvolta)
		tp = sum((df.convieneSvolta==True) & (isSvolta==True))
		tn = sum((df.convieneSvolta==False) & (isSvolta==False))
		fp = sum((df.convieneSvolta==False) & (isSvolta==True))
		fn = sum((df.convieneSvolta==True) & (isSvolta==False))
		
		#print(f"TP: {tp}\t TN: {tn}\t FP: {fp}\t FN: {fn}")
		accuracyList = accuracyList.append({'individual':p,'fp':fp,'fn':fn,'tp':tp,'tn':tn},ignore_index=True)
		'''print("True_y:")
		print(df.convieneSvolta.values)
		print("Pred_y:")
		print(isSvolta.values)'''
		#temp = np.vstack((df.convieneSvolta.values,isSvolta.values))
		temp = pd.DataFrame(list(zip(df.convieneSvolta.values,isSvolta.values)),columns=['true_y','pred_y'])
		#print(temp)
		accuracyList_pred = accuracyList_pred.append(temp, ignore_index=True)
		i = i + 1
		
	#print(accuracyList)
	save_accuracy(accuracyList,conf,None,conf.MAX_GENERATIONS)
	save_accuracy(accuracyList_pred,conf,"detailed_acccuracy.txt")
	#print(accuracyList_pred)
		
	
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