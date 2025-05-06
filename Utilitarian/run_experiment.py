import subprocess
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


end_value = 6
prob_value = (1.0, 0.8, 0.5, 0.1)

prob = 0.1
i = 2

#for i in range(1,end_value):
import subprocess

# Parametri
i = 4
prob = 0.1

# Comando corretto
command = [
    "python3",
    "./-25-KMEA-/fixedCode/ga_general.py",
    "-g", "100",
    "-p", "100",
    #"-a", f"0.{i}",
    "-r",
    "-e", str(prob),
    "-o", "-25-KMEA-/outputTest"
]

# Esecuzione
print("Eseguo:", " ".join(command))
result = subprocess.run(command)

# Controllo esito
print("Comando eseguito correttamente:", result.returncode == 0)


