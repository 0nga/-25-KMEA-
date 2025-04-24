import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


plt.figure(figsize=(8,8))

prob=1.0
java=False

for j in range(2,3):

	if java:
		path="/Users/aloreggia/Downloads/test/java/reward_completo_costPed1.0probPed_"+str(prob)+"_altruism_0."+str(j)
	else:
		path="/Users/aloreggia/Downloads/test/500ge/fixing_altruism_0."+str(j)+"_probPed_1.0"
		#path="/Users/aloreggia/Downloads/test/500ge/default"
		
	#path="/Users/aloreggia/Downloads/test/pythonTest_altruism_"+str(j)
	#path="/Users/aloreggia/Downloads/test/pythonTest_altruism_0."+str(j)+"_probPed_0.8"
	#path="/Users/aloreggia/Downloads/test/1000ge/tournament_tanh_reward_pythonTest_altruism_0."+str(j)+"_probPed_1.0_pop_100_gen_500"
	#path="/Users/aloreggia/Downloads/test/pythonTest.org"
	path="/Users/aloreggia/Downloads/test/500ge/evaluating_alternatives/correct_01/yescost_yesreward"
	
	pathLog="/logs/"

	altruismLevel= 2 
	probPed = None
	probPass=-1
	costPed = -1
	max_gen=1000
	pop_size=1000

	with open(path+'/out.txt', 'r') as f:
		wordcheck1="altruisticBehavior"
		wordcheck2="probDeathPedestrians"
		wordcheck3="costPedestrian"
		wordcheck4="MAX_GENERATIONS"
		wordcheck5="POPULATION_SIZE"
		wordcheck6="probDeathPassengers"
		for line in f:
			try:
				key, val = line.strip().split(': ')
				if wordcheck1 == key:
					#print(val)
					altruismLevel = val
				if wordcheck2 == key:
					#print(val)
					probPed = val
				if wordcheck3 == key:
					#print(val)
					costPed = val
				if wordcheck4 == key:
					#print(val)
					max_gen = eval(val)
				if wordcheck5 == key:
					#print(val)
					pop_size = eval(val)
				if wordcheck6 == key:
					#print(val)
					probPass = eval(val)
					#else:
					#	print ('notfound')
			except Exception:
				print("")

	sns.set(style="darkgrid")

	avgFitness=pd.DataFrame([])
	i=0
	'''for file in os.listdir(path):
		if file.endswith(".txt"):
			#print(os.path.join(path, file))
			l=pd.read_csv(os.path.join(path, file), delimiter="\t", decimal=",")
			l["type"]=i
			#print("Fitness: ",np.mean(l.Fitness))
			#print("Alrtruism: ",np.mean(l.AltruismLevel))
			avgFitness=avgFitness.append(l)
			i=i+1'''
	accuracyList=pd.DataFrame(columns=['generation','utilityV','minV','maxV','nnV'])
	i=0
	fp_list=[]
	tp_list=[]
	fn_list=[]
	tn_list=[]
	for i in range(max_gen):
		s="%03d"%i
		#print(os.path.join(path, file))
		migliaia=int(i/1000);
		l=pd.read_csv(os.path.join(path+pathLog+str(migliaia), "gen_"+s+".txt"), delimiter="\t", decimal=",")
		#print(l)
		l["type"]=i
		'''l['u_svolta']=l.numberOfPedestrians*l.AltruismLevel - l.numberOfPassengers * (1-l.AltruismLevel)
		l['u_dritto']=l.numberOfPassengers * (1-l.AltruismLevel) - l.numberOfPedestrians*l.AltruismLevel - l.numberOfPedestrians
		l['convieneSvolta']= l.u_svolta>l.u_dritto
		l['isSvolta']=(l.KnobLevel * probPass * l.numberOfPassengers < (1-l.KnobLevel) * probPed * l.numberOfPedestrians)'''
		
		accuracy=(l.convieneSvolta==l.predAction)

		utility_victims=sum(l.numberOfPedestrians[l.convieneSvolta==False])+sum(l.numberOfPassengers[l.convieneSvolta==True])
		min_victims= sum(l.loc[:,"numberOfPedestrians":"numberOfPassengers"].drop("probPed",axis=1).min(axis=1))
		nn_victims=sum(l.numberOfPedestrians[l.predAction==False])+sum(l.numberOfPassengers[l.predAction==True])
		max_victims= sum(l.loc[:,"numberOfPedestrians":"numberOfPassengers"].drop("probPed",axis=1).max(axis=1))
		
		#print(f"TP: {tp}\t TN: {tn}\t FP: {fp}\t FN: {fn}")
		accuracyList=accuracyList.append({'generation':i,'utilityV':utility_victims,'minV':min_victims,'maxV':max_victims,'nnV':nn_victims},ignore_index=True)
		
	plt.clf()
	
	print(accuracyList)

	ax=plt.plot(accuracyList.generation, accuracyList.utilityV, label="Utility Victims")
	plt.plot(accuracyList.generation, accuracyList.minV, label="Min Victims")
	plt.plot(accuracyList.generation, accuracyList.maxV, label="Max Victims")
	plt.plot(accuracyList.generation, accuracyList.nnV, label="NN Victims")
	plt.legend(loc="best")
	plt.xlabel('Generation')
	plt.ylabel('Number of Victims')
	#plt.title(title)
	plt.show()

	'''plt.clf()
	fig, ax1 = plt.subplots(figsize=(10,6))
	color = 'tab:green'
	ax1.set_xlabel('Generation', fontsize=16)
	ax1.set_ylabel('Number of Victims', fontsize=16, color=color)
	ax2 = sns.barplot(x='generation', y='nnV', data = accuracyList)
	ax1.tick_params(axis='y')
	ax2 = ax1.twinx()
	color = 'tab:red'
	#ax2.set_ylabel('Avg Percipitation %', fontsize=16, color=color)
	ax2 = sns.lineplot(x='generation', y='utilityV', data = accuracyList, sort=False, color=color)
	ax2.tick_params(axis='y', color=color)
	plt.show()'''
