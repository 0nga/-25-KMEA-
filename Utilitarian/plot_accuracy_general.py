import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

'''
Funzionano entrambi i grafici
'''

output_folder = "-25-KMEA-/grafici"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(8, 8))

# Parametri di test
list_test = ["reward_05_cost_0_1_bis"]
path = "/Users/onga/git/-25-KMEA-/outputTest/"
pathLog = "logs/"

for folder in list_test:
    altruismLevel = 2
    probPed = None
    probPass = -1
    costPed = -1
    max_gen = 20
    pop_size = 100

    with open(path + '/out.txt', 'r') as f:
        for line in f:
            try:
                key, val = line.strip().split(': ')
                if key == "altruisticBehavior":
                    altruismLevel = val
                if key == "probDeathPedestrians":
                    probPed = val
                if key == "costPedestrian":
                    costPed = val
                if key == "MAX_GENERATIONS":
                    max_gen = int(val)
                if key == "POPULATION_SIZE":
                    pop_size = int(val)
                if key == "probDeathPassengers":
                    probPass = float(val)
            except Exception:
                continue

    accuracyList = pd.DataFrame(columns=['generation', 'accuracy'])
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []

    for i in range(max_gen):
        s = f"{i:03d}"
        migliaia = int(i / 1000)
        l = pd.read_csv(os.path.join(path + pathLog + str(migliaia), f"gen_{s}.txt"), delimiter="\t", decimal=",")
        l["type"] = i
        
        accuracy = (l.convieneSvolta == l.predAction)
        
        tp = sum((l.convieneSvolta == True) & (l.predAction == True))
        tp_list.append(tp)
        tn = sum((l.convieneSvolta == False) & (l.predAction == False))
        tn_list.append(tn)
        fp = sum((l.convieneSvolta == False) & (l.predAction == True))
        fp_list.append(fp)
        fn = sum((l.convieneSvolta == True) & (l.predAction == False))
        fn_list.append(fn)

        accuracyList = pd.concat([accuracyList, pd.DataFrame([{'generation': i, 'accuracy': sum(accuracy)}])], ignore_index=True)

    # Verifica che accuracyList abbia dati validi
    accuracyList['generation'] = pd.to_numeric(accuracyList['generation'], errors='coerce')
    accuracyList['accuracy'] = pd.to_numeric(accuracyList['accuracy'], errors='coerce')
    accuracyList = accuracyList.dropna(subset=['generation', 'accuracy'])
    #print(accuracyList.head())
    
    # Traccia il grafico
    ax = sns.regplot(x="generation", y="accuracy", data=accuracyList, scatter=True)
    ax.set(xlabel='Generation', ylabel='Accuracy')
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, f"accuracy_{folder}.png"))
    plt.clf()
    
    # Confusion matrix plot
    ax = plt.plot(accuracyList.generation, np.array(accuracyList.accuracy) / pop_size, label="Accuracy")
    plt.plot(accuracyList.generation, np.array(tp_list) / pop_size, label="True Positive")
    plt.plot(accuracyList.generation, np.array(tn_list) / pop_size, label="True Negative")
    plt.plot(accuracyList.generation, np.array(fp_list) / pop_size, label="False Positive")
    plt.plot(accuracyList.generation, np.array(fn_list) / pop_size, label="False Negative")
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, f"conf_{folder}.png"))
