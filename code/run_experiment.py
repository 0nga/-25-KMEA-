import subprocess
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

end_value=6
prob_value=(1.0, 0.8, 0.5, 0.1)

prob=1.0

for i in range(1,end_value):
	command="-cp /Users/aloreggia/Dropbox/EUI/EthicalKnobSim-master/lib/*:/Users/aloreggia/Dropbox/EUI/EthicalKnobSim-master/bin/ simulator.SimulatorFactory /Users/aloreggia/Dropbox/EUI/EthicalKnobSim-master/config/config1d-directEncoding.txt /Users/aloreggia/Downloads/test/java 1 -c global --t 1 -d "+ str(prob)+" -a 0."+str(i)

	print(command)

	p=subprocess.Popen("java " + command, shell=True)
	print(p is None)

'''while p is None:
	j=0

plt.figure(figsize=(8,8))

for j in range(1,end_value):
	path="/Users/aloreggia/Downloads/test/newTest/completo_costPed1.0probPed_"+str(prob)+"_altruism_0."+str(j)
	pathLog="/logs/"

	altruismLevel= -1 
	probPed = -1
	costPed = -1

	with open(path+'/out.txt', 'r') as f:
		wordcheck1="altruisticBehavior"
		wordcheck2="probDeathPedestrians"
		wordcheck3="costPedestrian"
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
				#else:
				#	print ('notfound')
			except Exception:
				print("skipline")

	sns.set(style="darkgrid")

	avgFitness=pd.DataFrame([])
	i=0

	avgFitness=pd.DataFrame([])
	maxFitness=pd.DataFrame(columns=['index','maxFit'])
	i=0
	for i in range(1000):
		s="%03d"%i
		#print(os.path.join(path, file))
		migliaia=int(i/1000);
		l=pd.read_csv(os.path.join(path+pathLog+str(migliaia), "gen_"+s+".txt"), delimiter="\t", decimal=",")
		l["type"]=i
		#maxFitness=maxFitness.append(pd.DataFrame([[i,np.max(l.Fitness)]], columns=['index','maxFit']))
		#print("Fitness: ",np.mean(l.Fitness))
		#print("Alrtruism: ",np.mean(l.AltruismLevel))
		avgFitness=avgFitness.append(l)


	title="Probability of Death for Pedestrian: " + str(probPed)
	ax=sns.lineplot(x="type", y="KnobLevel", data=avgFitness, label="selfish: "+str(1-eval(altruismLevel)))
	ax.set(xlabel='Generation', ylabel='Knob Level')

	plt.ylim(0, 1.0)
	plt.title(title)

	#plt.show()
	plt.savefig("/Users/aloreggia/Dropbox/EUI/results/completo_costPed"+str(costPed)+"probPed"+str(probPed)+"altruism"+str(altruismLevel)+".png")'''
