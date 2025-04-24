import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


plt.figure(figsize=(8,8))

for j in range(1,2):
	#path="/Users/aloreggia/Downloads/test/newTest/completo_costPed1.0probPed_1.0_altruism_0."+str(j)
	path="/Users/aloreggia/Downloads/test/pythonTest/"
	pathLog="logs/0"

	altruismLevel= -1 
	probPed = -1
	costPed = -1

	with open(path+'/out.txt', 'r') as f:
		wordcheck1="altruisticBehavior"
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
	avgFitness=pd.DataFrame([])
	maxFitness=pd.DataFrame(columns=['index','maxFit'])
	i=0
	for i in range(max_gen):
		s="%03d"%i
		#print(os.path.join(path, file))
		l=pd.read_csv(os.path.join(path+pathLog, "gen_"+s+".txt"), delimiter="\t", decimal=",")
		l["type"]=i
		maxFitness=maxFitness.append(pd.DataFrame([[i,np.max(l.Fitness)]], columns=['index','maxFit']))
		#print("Fitness: ",np.mean(l.Fitness))
		#print("Alrtruism: ",np.mean(l.AltruismLevel))
		avgFitness=avgFitness.append(l)

	sns.lineplot(x="index", y="maxFit", data=maxFitness)
	plt.show()

	sns.lineplot(x="type", y="Fitness", data=avgFitness)
	plt.show()
