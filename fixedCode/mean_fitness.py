import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Directory dove salvare i grafici
output_dir = "/Users/onga/git/-25-KMEA-/grafici"
os.makedirs(output_dir, exist_ok=True)

probPed_list = [1, 0.8, 0.5, 0.3, 0.1]
avgFitness = pd.DataFrame([])

for j in range(1, 2):
    plt.figure(figsize=(8, 8))

    # Path dei dati
    path = "/Users/onga/git/-25-KMEA-/outputTest"
    pathLog = "logs/"

    altruismLevel = -1
    probPed = -1
    costPed = -1

    with open(os.path.join(path, 'out.txt'), 'r') as f:
        for line in f:
            try:
                key, val = line.strip().split(': ')
                if key == "ALTRUISM":
                    print(val)
                    altruismLevel = val
                elif key == "probDeathPedestrians":
                    print(val)
                    probPed = val
                elif key == "costPedestrian":
                    print(val)
                    costPed = val
                elif key == "MAX_GENERATIONS":
                    print(val)
                    max_gen = eval(val)
            except Exception:
                print("skipline")

    sns.set(style="darkgrid")
    print("Altruism: " + str(altruismLevel))
    avgFitness = pd.DataFrame([])
    maxFitness = pd.DataFrame(columns=['index', 'maxFit'])

    for i in range(max_gen):
        s = "%03d" % i
        migliaia = int(i / 1000)
        log_path = os.path.join(path, pathLog, str(migliaia), f"gen_{s}.txt")
        l = pd.read_csv(log_path, delimiter="\t", decimal=",")
        l["type"] = i

        avgFitness = pd.concat([avgFitness, l], ignore_index=True)

    title = "Probability of Death for Pedestrian: " + str(probPed)
    ax = sns.lineplot(x="type", y="AltruismLevel", data=avgFitness, label="selfish: " + str(1 - eval(altruismLevel)))
    ax.set(xlabel='Generation', ylabel='AltruismLevel')

    plt.ylim(0, 1.0)
    plt.title(title)

    # Nome file dinamico
    filename = f"altruism_probPed_{probPed}_costPed_{costPed}.png"
    filepath = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Salvato: {filepath}")
