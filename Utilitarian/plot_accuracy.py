import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(8, 8))

# === Configurazioni iniziali ===
prob = 1.0
java = False
path = "/Users/onga/git/-25-KMEA-/outputTest/"
pathLog = "logs/"
graphPath = "/Users/onga/git/-25-KMEA-/"
output_dir = os.path.join(graphPath, "grafici")
os.makedirs(output_dir, exist_ok=True)

# === Lettura dei parametri da out.txt ===
altruismLevel = -1
probPed = -1
probPass = -1
costPed = -1
max_gen = 50
pop_size = 100

with open(os.path.join(path, 'out.txt'), 'r') as f:
    for line in f:
        try:
            key, val = line.strip().split(': ')
            if key == "altruisticBehavior":
                altruismLevel = val
            elif key == "probDeathPedestrians":
                probPed = val
            elif key == "costPedestrian":
                costPed = val
            elif key == "MAX_GENERATIONS":
                max_gen = eval(val)
            elif key == "POPULATION_SIZE":
                pop_size = eval(val)
            elif key == "probDeathPassengers":
                probPass = eval(val)
        except Exception:
            print("skipline")

# === Inizializzazione dataframe ===
sns.set(style="darkgrid")
accuracy_data = []
tp_list, tn_list, fp_list, fn_list = [], [], [], []

# === Elaborazione per generazione ===
for i in range(max_gen):
    s = f"{i:03d}"
    migliaia = i // 1000
    gen_path = os.path.join(path, pathLog, str(migliaia), f"gen_{s}.txt")

    try:
        l = pd.read_csv(gen_path, delimiter="\t", decimal=",")
        l["type"] = i

        # Calcolo accuratezza
        accuracy = (l.convieneSvolta == l.predAction)
        accuracy_data.append({'generation': i, 'accuracy': sum(accuracy)})

        # TP / TN / FP / FN
        tp = sum((l.convieneSvolta == True) & (l.predAction == True))
        tn = sum((l.convieneSvolta == False) & (l.predAction == False))
        fp = sum((l.convieneSvolta == False) & (l.predAction == True))
        fn = sum((l.convieneSvolta == True) & (l.predAction == False))
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)

        # === ROC per ogni generazione ===
        y_true = l.convieneSvolta.astype(int)
        y_score = l.predAction.astype(float)

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, marker='.', label=f'Gen {i} (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Generation {i}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"roc_gen_{i}.png"))
            plt.close()
        else:
            print(f"⚠️ ROC non calcolabile per generazione {i}: solo una classe in y_true → {np.unique(y_true)}")

    except Exception as e:
        print(f"Errore alla generazione {i}: {e}")

# === Curve ROC complessiva sull’ultima generazione ===
y_true = l.convieneSvolta.astype(int)
y_score = l.predAction.astype(float)

if len(np.unique(y_true)) > 1:
    ns_probs = np.random.randint(2, size=len(y_true))
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='NN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title(f"ROC Curve - Ultima Generazione (AUC = {roc_auc_score(y_true, y_score):.2f})")
    plt.savefig(os.path.join(output_dir, f"roc_curve_altruism_{altruismLevel}.png"))
    plt.close()

# === Accuracy ===
accuracyList = pd.DataFrame(accuracy_data)
accuracyList.generation = accuracyList.generation.astype('float64')
accuracyList.accuracy = accuracyList.accuracy.astype('float64')

title = f"{'JAVA' if java else 'PYTHON'} - Death Pedestrian: {probPed} selfish: {1 - eval(altruismLevel)}"

# Regressione accuracies
ax = sns.regplot(
    x="generation",
    y="accuracy",
    data=accuracyList.assign(accuracy=accuracyList.accuracy / pop_size),
    label=f"selfish: {1 - eval(altruismLevel)}",
    scatter=True
)
ax.legend(loc=4)
ax.set(xlabel='Generation', ylabel='Accuracy')
plt.title(title)
plt.savefig(os.path.join(output_dir, f"accuracy_curve_altruism_{altruismLevel}.png"))
plt.clf()

# Accuratezza e metriche TP, TN, FP, FN
plt.plot(accuracyList.generation, np.array(accuracyList.accuracy) / pop_size, label="Accuracy")
plt.plot(accuracyList.generation, np.array(tp_list) / pop_size, label="True Positive")
plt.plot(accuracyList.generation, np.array(tn_list) / pop_size, label="True Negative")
plt.plot(accuracyList.generation, np.array(fp_list) / pop_size, label="False Positive")
plt.plot(accuracyList.generation, np.array(fn_list) / pop_size, label="False Negative")
plt.legend(loc="best")
plt.title(title)
plt.savefig(os.path.join(output_dir, f"accuracy_curve_altruism2_{altruismLevel}.png"))
plt.clf()
