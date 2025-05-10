import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix,precision_score, recall_score, f1_score
import argparse
import shutil  # Import the shutil module for directory deletion

# Elimina la directory grafici prima della creazione dei nuovi grafici
def setup_output_directory(script_dir):
    output_dir = os.path.join(script_dir, "grafici")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete the directory and its contents

    os.makedirs(output_dir, exist_ok=False)  # Recreate the directory

    return output_dir

def read_parameters(path, max_gen=None):
    params = {
        'altruismLevel': -1,
        'probPed': -1,
        'probPass': -1,
        'costPed': -1,
        'max_gen': 50,  # Default value for max_gen
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
        params['max_gen'] = max_gen  # Override with the value passed via command line

    return params

def get_log_path(base_path, generation):
    s = f"{generation:03d}"
    migliaia = generation // 1000
    return os.path.join(base_path, "logs", str(migliaia), f"gen_{s}.txt")

def plot_max_fitness(path, output_dir, params):
    sns.set(style="darkgrid")
    maxFitness = pd.DataFrame(columns=['index', 'maxFit'])
    avgFitness = pd.DataFrame([])

    for i in range(params['max_gen']):
        filepath = get_log_path(path, i)
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
        log_path = get_log_path(path, i)
        l = pd.read_csv(log_path, delimiter="\t", decimal=",")
        l["type"] = i
        avgFitness = pd.concat([avgFitness, l], ignore_index=True)

    plt.figure(figsize=(8, 8))
    title = f"Probability of Death for Pedestrian: {params['probPed']}"
    altruism_value = float(params['altruismLevel']) if isinstance(params['altruismLevel'], str) else params['altruismLevel']
    ax = sns.lineplot(x="type", y="AltruismLevel", data=avgFitness, 
                     label=f"selfish: {1 - altruism_value}")
    ax.set(xlabel='Generation', ylabel='AltruismLevel')
    plt.title(title)

    filename = f"altruism_probPed_{params['probPed']}_costPed_{params['costPed']}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_accuracy(path, output_dir, params):
    accuracy_data = []
    tp_list, tn_list, fp_list, fn_list = [], [], [], []

    for i in range(params['max_gen']):
        gen_path = get_log_path(path, i)

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
        # Decommenta per avere le ROC-AUC di ogni gen
        '''ns_probs = np.random.randint(2, size=len(y_true))
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
        plt.close()'''

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

    # Prepare lists for true and predicted labels
    y_true = []
    y_pred = []

    for i in range(params['max_gen']):
        gen_path = get_log_path(path, i)

        try:
            l = pd.read_csv(gen_path, delimiter="\t", decimal=",")
            l["type"] = i

            # True labels
            y_true.extend(l.convieneSvolta.astype(int))  # Assuming 'convieneSvolta' is the true label
            # Predicted labels
            y_pred.extend(l.predAction.astype(int))  # Assuming 'predAction' is the predicted label

        except Exception as e:
            print(f"Errore alla generazione {i}: {e}")

def plot_classification_metrics(path, output_dir, params):
    precision_list = []
    recall_list = []
    f1_list = []
    f2_list = []

    for i in range(params['max_gen']):
        gen_path = get_log_path(path, i)
        try:
            l = pd.read_csv(gen_path, delimiter="\t", decimal=",")
            y_true = l.convieneSvolta.astype(int)
            y_pred = l.predAction.astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0

            precision_list.append({'generation': i, 'precision': precision})
            recall_list.append({'generation': i, 'recall': recall})
            f1_list.append({'generation': i, 'f1_score': f1})
            f2_list.append({'generation': i, 'f2_score': f2})

        except Exception as e:
            print(f"Errore alla generazione {i}: {e}")

    altruism_value = float(params['altruismLevel']) if isinstance(params['altruismLevel'], str) else params['altruismLevel']
    title = f"Death Pedestrian: {params['probPed']} selfish: {1 - altruism_value}"

    def plot_metric(data_list, metric_name, ylabel):
        df = pd.DataFrame(data_list)
        df.generation = df.generation.astype('float64')
        df[metric_name] = df[metric_name].astype('float64')

        plt.figure(figsize=(8, 8))
        ax = sns.regplot(
            x="generation",
            y=metric_name,
            data=df,
            scatter=True,
            label=f"selfish: {1 - altruism_value}"
        )
        ax.legend(loc=4)
        ax.set(xlabel='Generation', ylabel=ylabel)
        plt.title(title)
        plt.savefig(os.path.join(output_dir, f"{metric_name}_curve_altruism_{params['altruismLevel']}.png"))
        plt.close()

    plot_metric(precision_list, 'precision', 'Precision')
    plot_metric(recall_list, 'recall', 'Recall')
    plot_metric(f1_list, 'f1_score', 'F1 Score')
    plot_metric(f2_list, 'f2_score', 'F2 Score')

def plot_confusion_matrix_last_generation(path, output_dir, params):
    # Get the path for the last generation
    gen_path = get_log_path(path, params['max_gen'] - 1)  # Ultima generazione

    try:
        # Read the log file for the last generation
        l = pd.read_csv(gen_path, delimiter="\t", decimal=",")
        l["type"] = params['max_gen'] - 1  # Assign the generation number

        # True labels
        y_true = l.convieneSvolta.astype(int)  # Assuming 'convieneSvolta' is the true label
        # Predicted labels
        y_pred = l.predAction.astype(int)  # Assuming 'predAction' is the predicted label

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred No', 'Pred Yes'], yticklabels=['True No', 'True Yes'])
        plt.title(f"Confusion Matrix - Generation {params['max_gen']}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_last_gen_{params['max_gen']}.png"))
        plt.close()

    except Exception as e:
        print(f"Errore durante la generazione della matrice di confusione per l'ultima generazione: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generations", type=int, default=50, help="Number of generations to read")  # Modifica qui
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "outputTest")
    output_dir = setup_output_directory(script_dir)

    params = read_parameters(data_path, args.generations)  # Passiamo il parametro delle generazioni

    plot_max_fitness(data_path, output_dir, params)
    plot_mean_fitness(data_path, output_dir, params)
    plot_accuracy(data_path, output_dir, params)
    plot_confusion_matrix_last_generation(data_path, output_dir, params)
    plot_classification_metrics(data_path, output_dir, params)

if __name__ == "__main__":
    main()
