import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import argparse

def setup_output_directory(script_dir):
    """Crea e restituisce la directory per l'output dei grafici."""
    output_dir = os.path.join(script_dir, "grafici")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def read_parameters(path, max_gen=None):
    """Legge i parametri da out.txt nella cartella outputTest"""
    params = {
        'altruismLevel': -1,
        'probPed': -1,
        'probPass': -1,
        'costPed': -1,
        'max_gen': 20,
        'pop_size': 100
    }

    with open(os.path.join(path, 'out.txt'), 'r') as f:
        for line in f:
            try:
                key, val = line.strip().split(': ')
                if key == "altruisticBehavior":
                    params['altruismLevel'] = val
                elif key == "probDeathPedestrians":
                    params['probPed'] = val
                elif key == "costPedestrian":
                    params['costPed'] = val
                elif key == "MAX_GENERATIONS":
                    params['max_gen'] = eval(val)
                elif key == "POPULATION_SIZE":
                    params['pop_size'] = eval(val)
                elif key == "probDeathPassengers":
                    params['probPass'] = val
            except Exception:
                continue

    if max_gen is not None:
        params['max_gen'] = max_gen

    return params

def plot_max_fitness(path, output_dir, params):
    sns.set(style="darkgrid")
    maxFitness = pd.DataFrame(columns=['index', 'maxFit'])
    avgFitness = pd.DataFrame([])

    for i in range(params['max_gen']):
        s = "%03d" % i
        migliaia = int(i / 1000)
        filepath = os.path.join(path, "logs", str(migliaia), f"gen_{s}.txt")
        l = pd.read_csv(filepath, delimiter="\t", decimal=",")
        l["type"] = i

        maxFitness = pd.concat([maxFitness, pd.DataFrame([[i, np.max(l.Fitness)]], columns=['index', 'maxFit'])], ignore_index=True)
        avgFitness = pd.concat([avgFitness, l], ignore_index=True)

    plt.figure(figsize=(8, 8))
    sns.lineplot(x="index", y="maxFit", data=maxFitness)
    plt.title("Andamento della Massima Fitness")
    plt.xlabel("Generazione")
    plt.ylabel("Fitness Massima")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maxFit.png"))
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.lineplot(x="type", y="Fitness", data=avgFitness)
    plt.title("Fitness Media per Generazione")
    plt.xlabel("Generazione")
    plt.ylabel("Fitness Media")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avgFit.png"))
    plt.close()

def plot_mean_fitness(path, output_dir, params):
    sns.set(style="darkgrid")
    avgFitness = pd.DataFrame([])

    for i in range(params['max_gen']):
        s = "%03d" % i
        migliaia = int(i / 1000)
        log_path = os.path.join(path, "logs", str(migliaia), f"gen_{s}.txt")
        l = pd.read_csv(log_path, delimiter="\t", decimal=",")
        l["type"] = i
        avgFitness = pd.concat([avgFitness, l], ignore_index=True)

    plt.figure(figsize=(8, 8))
    title = f"Probability of Death for Pedestrian: {params['probPed']}"
    altruism_value = float(params['altruismLevel']) if isinstance(params['altruismLevel'], str) else params['altruismLevel']
    ax = sns.lineplot(x="type", y="AltruismLevel", data=avgFitness, 
                     label=f"selfish: {1 - altruism_value}")
    ax.set(xlabel='Generation', ylabel='AltruismLevel')
    plt.ylim(0, 1.0)
    plt.title(title)

    filename = f"altruism_probPed_{params['probPed']}_costPed_{params['costPed']}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_accuracy(path, output_dir, params):
    accuracy_data = []
    tp_list, tn_list, fp_list, fn_list = [], [], [], []

    for i in range(params['max_gen']):
        s = f"{i:03d}"
        migliaia = i // 1000
        gen_path = os.path.join(path, "logs", str(migliaia), f"gen_{s}.txt")

        try:
            l = pd.read_csv(gen_path, delimiter="\t", decimal=",")
            l["type"] = i

            accuracy = (l.convieneSvolta == l.predAction)
            accuracy_data.append({'generation': i, 'accuracy': sum(accuracy)})

            tp = sum((l.convieneSvolta == True) & (l.predAction == True))
            tn = sum((l.convieneSvolta == False) & (l.predAction == False))
            fp = sum((l.convieneSvolta == False) & (l.predAction == True))
            fn = sum((l.convieneSvolta == True) & (l.predAction == False))

            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)

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

        except Exception as e:
            print(f"Errore alla generazione {i}: {e}")

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
        plt.savefig(os.path.join(output_dir, f"roc_curve_altruism_{params['altruismLevel']}.png"))
        plt.close()

    accuracyList = pd.DataFrame(accuracy_data)
    accuracyList.generation = accuracyList.generation.astype('float64')
    accuracyList.accuracy = accuracyList.accuracy.astype('float64')

    altruism_value = float(params['altruismLevel']) if isinstance(params['altruismLevel'], str) else params['altruismLevel']
    title = f"Death Pedestrian: {params['probPed']} selfish: {1 - altruism_value}"

    plt.figure(figsize=(8, 8))
    ax = sns.regplot(
        x="generation",
        y="accuracy",
        data=accuracyList.assign(accuracy=accuracyList.accuracy / params['pop_size']),
        label=f"selfish: {1 - altruism_value}",
        scatter=True
    )
    ax.legend(loc=4)
    ax.set(xlabel='Generation', ylabel='Accuracy')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f"accuracy_curve_altruism_{params['altruismLevel']}.png"))
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(accuracyList.generation, np.array(accuracyList.accuracy) / params['pop_size'], label="Accuracy")
    plt.plot(accuracyList.generation, np.array(tp_list) / params['pop_size'], label="True Positive")
    plt.plot(accuracyList.generation, np.array(tn_list) / params['pop_size'], label="True Negative")
    plt.plot(accuracyList.generation, np.array(fp_list) / params['pop_size'], label="False Positive")
    plt.plot(accuracyList.generation, np.array(fn_list) / params['pop_size'], label="False Negative")
    plt.legend(loc="best")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f"accuracy_curve_altruism2_{params['altruismLevel']}.png"))
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Genera i grafici per il risultato del GA')
    parser.add_argument('-g', '--generations', type=int, help='Numero di generazioni da elaborare (sovrascrive out.txt)')
    args = parser.parse_args()

    data_path = os.path.join(script_dir, "outputTest")
    output_dir = setup_output_directory(script_dir)

    params = read_parameters(data_path, args.generations)
    plot_max_fitness(data_path, output_dir, params)
    plot_mean_fitness(data_path, output_dir, params)
    plot_accuracy(data_path, output_dir, params)

    print("Tutti i grafici sono stati generati con successo!")

if __name__ == "__main__":
    main()