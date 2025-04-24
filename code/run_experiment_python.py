import subprocess
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

end_value=6
prob_value=(1.0, 0.8, 0.5, 0.3, 0.1)

prob=1.0

for probPed in prob_value:
	for i in range(1,end_value):
		command="/Users/aloreggia/Dropbox/EUI/EthicalKnobSim-master/scripts/processing/ga_python_conf_general.py  -g 500 -p 100 -a 0."+str(i) +" -e "+str(probPed)+ " -o /Users/aloreggia/Downloads/test/500ge/fixing_altruism_0."+str(i)+"_probPed_"+str(probPed)

		print(command)

		p=subprocess.run("python " + command, shell=True)
		print(p == None)

