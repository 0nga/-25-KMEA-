import subprocess
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

end_value=6
prob_value=(1.0, 0.8, 0.5, 0.1)

prob=1.0

command = "./-25-KMEA-/code/ga_python_conf_general.py -g 50 -p 100 -r -e " + str(prob) + " -o -25-KMEA-/code/outputTest"

print(command)

p=subprocess.run("python3 " + command, shell=True)
print(p == None)

