import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Percorso assoluto della directory in cui salvare i grafici
output_dir = "/Users/onga/git/-25-KMEA-/grafici"
os.makedirs(output_dir, exist_ok=True)

for j in range(1, 2):
    path = "/Users/onga/git/-25-KMEA-/outputTest/"
    pathLog = "logs/0"

    altruismLevel = -1
    probPed = -1
    costPed = -1

    with open(os.path.join(path, 'out.txt'), 'r') as f:
        for line in f:
            try:
                key, val = line.strip().split(': ')
                if key == "altruisticBehavior":
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
    avgFitness = pd.DataFrame([])
    maxFitness = pd.DataFrame(columns=['index', 'maxFit'])

    for i in range(max_gen):
        s = "%03d" % i
        filepath = os.path.join(path, pathLog, f"gen_{s}.txt")
        l = pd.read_csv(filepath, delimiter="\t", decimal=",")
        l["type"] = i

        maxFitness = pd.concat([maxFitness, pd.DataFrame([[i, np.max(l.Fitness)]], columns=['index', 'maxFit'])], ignore_index=True)
        avgFitness = pd.concat([avgFitness, l], ignore_index=True)

    # Salva grafico maxFitness
    plt.figure(figsize=(8, 8))
    sns.lineplot(x="index", y="maxFit", data=maxFitness)
    plt.title("Andamento della Massima Fitness")
    plt.xlabel("Generazione")
    plt.ylabel("Fitness Massima")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maxFit.png"))
    plt.close()

    # (opzionale) Salva anche grafico medio fitness
    plt.figure(figsize=(8, 8))
    sns.lineplot(x="type", y="Fitness", data=avgFitness)
    plt.title("Fitness Media per Generazione")
    plt.xlabel("Generazione")
    plt.ylabel("Fitness Media")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avgFit.png"))
    plt.close()
