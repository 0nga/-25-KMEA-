import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


probPed_list=[1, 0.8, 0.5, 0.3, 0.1]
avgFitness=pd.DataFrame([])

for j in range(1,2):
	plt.figure(figsize=(8,8))
	#path="/Users/aloreggia/Downloads/test/newTest/completo_costPed1.0probPed_1.0_altruism_0."+str(j)
	#path="/Users/aloreggia/Downloads/test/pythonTest_altruism_"+str(j)
	#path="/Users/aloreggia/Downloads/test/1000ge/tournament_tanh_reward_pythonTest_altruism_0."+str(j)+"_probPed_1.0_pop_100_gen_500"
	#path="/Users/aloreggia/Downloads/test/500ge/fixing_altruism_0."+str(j)+"_probPed_0.3"
	path="/Users/aloreggia/Downloads/test/500ge/evaluating_alternatives/correct_01/nocost_yesreward"
	pathLog="/logs/"

	altruismLevel= -1 
	probPed = -1
	costPed = -1

	with open(path+'/out.txt', 'r') as f:
		wordcheck1="ALTRUISM"
		wordcheck2="probDeathPedestrians"
		wordcheck3="costPedestrian"
		wordcheck4="MAX_GENERATIONS"
		for line in f:
			try:
				key, val = line.strip().split(': ')
				if wordcheck1 == key:
					print(val)
					altruismLevel = val
				if wordcheck2 == key:
					print(val)
					probPed = val
				if wordcheck3 == key:
					print(val)
					costPed = val
				if wordcheck4 == key:
					print(val)
					max_gen = eval(val)
				#else:
				#	print ('notfound')
			except Exception:
				print("skipline")

	sns.set(style="darkgrid")
	print("Altruism: "+ str(altruismLevel))
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
	maxFitness=pd.DataFrame(columns=['index','maxFit'])
	i=0
	for i in range(max_gen):
		s="%03d"%i
		#print(os.path.join(path, file))
		migliaia=int(i/1000);
		l=pd.read_csv(os.path.join(path+pathLog+str(migliaia), "gen_"+s+".txt"), delimiter="\t", decimal=",")
		l["type"]=i
		#maxFitness=maxFitness.append(pd.DataFrame([[i,np.max(l.Fitness)]], columns=['index','maxFit']))
		#print(str(i)+" Fitness: ",np.mean(l.Fitness))
		#print("Alrtruism: ",np.mean(l.AltruismLevel))
		avgFitness=avgFitness.append(l)


	title="Probability of Death for Pedestrian: " + str(probPed)
	ax=sns.lineplot(x="type", y="AltruismLevel", data=avgFitness, label="selfish: "+str(1-eval(altruismLevel)))
	ax.set(xlabel='Generation', ylabel='AltruismLevel')
	
	plt.ylim(0, 1.0)
	plt.title(title)

	plt.show()
	#plt.savefig("/Users/aloreggia/Dropbox/EUI/results/completo_costPed"+str(costPed)+"probPed"+str(probPed)+"altruism"+str(altruismLevel)+".png")
	#plt.savefig("/Users/aloreggia/Dropbox/EUI/results/completo_costPed"+str(costPed)+"probPed"+str(probPed)+"altruism"+str(altruismLevel)+".png")
