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

for j in range(1,6):
	if java:
		path="/Users/aloreggia/Downloads/test/java/reward_completo_costPed1.0probPed_"+str(prob)+"_altruism_0."+str(j)
	else:
		path="/Users/aloreggia/Downloads/test/500ge/tournament_tanh_reward_pythonTest_altruism_0."+str(j)+"_probPed_1.0_pop_100_gen_500"
		
	#path="/Users/aloreggia/Downloads/test/pythonTest_altruism_"+str(j)
	#path="/Users/aloreggia/Downloads/test/pythonTest_altruism_0."+str(j)+"_probPed_0.8"
	#path="/Users/aloreggia/Downloads/test/1000ge/tournament_tanh_reward_pythonTest_altruism_0."+str(j)+"_probPed_1.0_pop_100_gen_500"
	#path="/Users/aloreggia/Downloads/test/pythonTest.org"
	
	path = "/Users/onga/git/TestOut/"
	pathLog = "logs/"	

	altruismLevel= -1 
	probPed = -1
	probPass=-1
	costPed = -1
	max_gen=50
	pop_size=100

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
				if wordcheck5 == key:
					print(val)
					pop_size = eval(val)
				if wordcheck6 == key:
					print(val)
					probPass = eval(val)
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
	#accuracyList=pd.DataFrame(columns=['generation','accuracy'])
	accuracy_data = []

	maxFitness=pd.DataFrame(columns=['index','maxFit'], dtype=float)
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
		
		tp = sum((l.convieneSvolta==True) & (l.predAction==True))
		tp_list.append(tp)
		tn = sum((l.convieneSvolta==False) & (l.predAction==False))
		tn_list.append(tn)
		fp = sum((l.convieneSvolta==False) & (l.predAction==True))
		fp_list.append(fp)
		fn = sum((l.convieneSvolta==True) & (l.predAction==False))
		fn_list.append(fn)
		
		#print(f"TP: {tp}\t TN: {tn}\t FP: {fp}\t FN: {fn}")
		#accuracyList=accuracyList.append({'generation':i,'accuracy':sum(accuracy)},ignore_index=True)
		accuracy_data.append({'generation': i, 'accuracy': sum(accuracy)})

		

	ns_probs = np.random.randint(2, size=len(l.predAction))
	ns_fpr, ns_tpr, _ = roc_curve(l.convieneSvolta, ns_probs)

	lr_fpr, lr_tpr, _ = roc_curve(l.convieneSvolta, l.predAction)
	plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
	plt.plot(lr_fpr, lr_tpr, marker='.', label='NN')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()
	plt.show()

	accuracyList = pd.DataFrame(accuracy_data)

	accuracyList.generation=accuracyList.generation.astype('float64')
	accuracyList.accuracy=accuracyList.accuracy.astype('float64')
	print(accuracyList)
	if java:
		title="JAVA - Death Pedestrian: " + str(probPed) + " selfish: " + str(1-eval(altruismLevel))
	else:
		title="PYTHON - Death Pedestrian: " + str(probPed) + " selfish: " + str(1-eval(altruismLevel))
#	ax=sns.regplot(accuracyList.generation, np.array(accuracyList.accuracy)/pop_size, data=accuracyList,label="selfish: "+str(1-eval(altruismLevel)), scatter=True)
	ax = sns.regplot(x="generation", y="accuracy", data=accuracyList.assign(accuracy=accuracyList.accuracy/pop_size), label="selfish: "+str(1-eval(altruismLevel)), scatter=True)
	ax.legend(loc=4)
	
	#ax=plt.plot(accuracyList.generation, accuracyList.accuracy)
	ax.set(xlabel='Generation', ylabel='Accuracy')
	
	#plt.ylim(0, 1.0)
	plt.title(title)

	plt.show()
	
	'''if java:
		plt.savefig("/Users/aloreggia/Dropbox/EUI/results/simulazione/simulazione_java/sns_accuracy_costPed"+str(costPed)+"probPed"+str(probPed)+"altruism"+str(altruismLevel)+"_gen_500.png")
	else:
		plt.savefig("/Users/aloreggia/Dropbox/EUI/results/simulazione/simulazione_python/sns_accuracy_costPed"+str(costPed)+"probPed"+str(probPed)+"altruism"+str(altruismLevel)+"_gen_500.png")'''
		
	plt.clf()
	
	ax=plt.plot(accuracyList.generation, np.array(accuracyList.accuracy)/pop_size, label="Accuracy")
	plt.plot(accuracyList.generation, np.array(tp_list)/pop_size, label="True Positive")
	plt.plot(accuracyList.generation, np.array(tn_list)/pop_size, label="True Negative")
	plt.plot(accuracyList.generation, np.array(fp_list)/pop_size, label="False Positive")
	plt.plot(accuracyList.generation, np.array(fn_list)/pop_size, label="False Negative")
	plt.legend(loc="best")
	plt.title(title)
	plt.show()
	
	#ax=plt.plot(accuracyList.generation, accuracyList.accuracy)
	#ax.set(xlabel='Generation', ylabel='Accuracy')
	#legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
	
	#plt.ylim(0, 1.0)
	
	'''if java:
		#plt.savefig("/Users/aloreggia/Dropbox/EUI/results/simulazione/simulazione_java/accuracy_costPed"+str(costPed)+"probPed"+str(probPed)+"altruism"+str(altruismLevel)+"_gen_500.png")
		plt.savefig("/Users/aloreggia/Dropbox/EUI/results/simulazione/simulazione_java/random_probs.png")
	else:
		plt.savefig("/Users/aloreggia/Dropbox/EUI/results/simulazione/simulazione_python/accuracy_costPed"+str(costPed)+"probPed"+str(probPed)+"altruism"+str(altruismLevel)+"_gen_500.png")'''
	plt.clf()
