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
	command="/Users/aloreggia/Dropbox/EUI/EthicalKnobSim-master/scripts/processing/ga_python_conf_fixingInd.py  -g 500 -p 100 -a 0."+str(i) +" -e "+str(prob)#+ " -o /Users/aloreggia/Downloads/test/1000ge/pythonTest_altruism_0."+str(i)+"_probPed_1.0"

	print(command)

	p=subprocess.run("python " + command, shell=True)
	print(p == None)

