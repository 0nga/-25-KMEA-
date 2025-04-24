import subprocess
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

end_value=6
prob_value=(1.0, 0.8, 0.5, 0.1)

prob=1.0


command="/Users/aloreggia/Dropbox/EUI/EthicalKnobSim-master/scripts/processing/ga_python_conf_general.py  -g 500 -p 100 -r -o /Users/aloreggia/Downloads/test/500ge/reward_05_cost_0_1_bis"

print(command)

p=subprocess.run("python " + command, shell=True)
print(p == None)

