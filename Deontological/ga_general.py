import os
import random
import pandas as pd
import numpy as np
import time
from keras.models import Sequential,Model # Keras è usato per la NN
from keras.layers import Dense, Dropout, InputLayer, Input
from keras import backend as K
from Configuration import Configuration # Importa la classe di configurazione
import copy

from Individual import * # Importa la classe Individual e le funzioni associate

import sys, getopt # Per gestire gli argomenti da riga di comando

import random
random.seed(42)
	
def main():
	"""
	Funzione principale che esegue l'algoritmo genetico per l'evoluzione
	di strategie di guida autonoma in scenari di dilemma etico.
	"""

	# Crea l'oggetto che conterrà la configurazione
	conf = Configuration() 

	# Gestione dei flag passati nel comando
	try:
		# Definisci gli argomenti attesi (h per help, a per altruism, ecc.)
		opts, args = getopt.getopt(sys.argv[1:], "ha:o:g:p:e:r", 
									["help", "altruism=", "output=", "gen=", "population=", "probped=", "randomizealtruism"])
	except getopt.GetoptError as err:
		print(f"Error parsing options: {err}")
		# Stampa un messaggio di aiuto se gli argomenti non sono validi
		print("Usage: ga_general.py -g <numGenerations> -p <numPopulation> -e <probDeathPedestrians> [-r] [-a <altruismLevel>] [-o <outputPath>]")
		sys.exit(2)

	# Elabora gli argomenti passati
	# print(f"Arguments received: {opts}") # Debug
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("Usage: ga_general.py -g <numGenerations> -p <numPopulation> -e <probDeathPedestrians> [-r] [-a <altruismLevel>] [-o <outputPath>]")
			sys.exit()
		elif opt in ("-a", "--altruism"):	
			conf.set_altruism(float(arg))
			print(f"Configuration: ALTRUISM set to {conf.ALTRUISM}")
		elif opt in ("-g", "--gen"):
			conf.MAX_GENERATIONS = int(arg)
			print(f"Configuration: MAX_GENERATIONS set to {conf.MAX_GENERATIONS}")
		elif opt in ("-p", "--population"):
			conf.set_population_size(int(arg)) # Questo metodo dovrebbe aggiornare anche BEST_CANDIDATES_COUNT ecc.
			print(f"Configuration: POPULATION_SIZE set to {conf.POPULATION_SIZE}")
		elif opt in ("-o", "--output"):	
			conf.set_path(arg) # Questo metodo imposta il percorso di output
			print(f"Configuration: Output path set to {conf.path}")
		elif opt in ("-e", "--probped"): # Probabilità di morte dei pedoni (se non randomizzata per scenario)
			conf.probDeathPedestrians = float(arg)
			print(f"Configuration: Default probDeathPedestrians set to {conf.probDeathPedestrians}")
		elif opt in ("-r", "--randomizealtruism"): # Flag per randomizzare l'altruismo degli individui	
			conf.randomizeAltruism = True
			print(f"Configuration: Individual altruism will be randomized.")
		
	# Salvataggio delle opzioni di configurazione correnti
	save_options(conf) # Funzione definita in Individual.py
		
	# >>>>>> Inizio Algoritmo Genetico <<<<<<
	print("\n********** Genetic Algorithm Starting **********")

	# Genera la prima popolazione di individui casualmente
	population_new = generate_first_population_randomly(conf) # Funzione da Individual.py
	
	# Ciclo principale dell'algoritmo genetico per ogni generazione
	for current_generation in range(conf.MAX_GENERATIONS):
		population = population_new # La nuova popolazione diventa quella corrente
		print(f"\n[+] Generation {current_generation+1} of {conf.MAX_GENERATIONS}")
		
		# DataFrame per i risultati della generazione corrente
		df_results_current_gen = pd.DataFrame(columns=[
			"numberOfPedestrians", "probPed", 
			"numberOfPassengers", "probPass",
			"AltruismLevel", "KnobLevel", "Fitness",
			'convieneSvolta', "predAction"
		])
		
		# Crea gli scenari per la popolazione corrente
		# La funzione createScenarios (da Individual.py) genera un DataFrame di scenari
		df_scenari_input_for_nn = createScenarios(conf, population, conf.randomizeAltruism)

		# Standardizza gli scenari di input per la rete neurale
		# Lo scaler viene fittato sui dati della generazione corrente.
		# È importante che la standardizzazione sia consistente se si confrontano knob/fitness tra generazioni.
		scenari_standardized_np, scaler = standardize(df_scenari_input_for_nn.copy()) # standardize da Individual.py

		# >>>>>> Fase di Valutazione della Popolazione <<<<<<
		print(f"\tEvaluating Population ({len(population)} individuals): ", end='', flush=True)
		evaluation_start_time = time.time()

		# Liste per raccogliere i dati dalla valutazione
		predAction_list_current_gen = []
		fitness_list_current_gen = []
		knob_list_current_gen = []

		# Valuta ogni individuo nella popolazione
		for i, individual_p in enumerate(population):
			# Prendi lo scenario standardizzato per l'individuo corrente
			# Assicurati che l'indicizzazione sia corretta
			if scenari_standardized_np.shape[0] > i:
				current_scenario_std_for_individual = np.array(scenari_standardized_np[i]).reshape((1,5))
				
				# Calcola fitness e azione predetta (l'azione sarà sempre 0 "dritto" a causa della modifica deontologica)
				# Lo scaler è passato per permettere la denormalizzazione all'interno di computeFitness.
				action = individual_p.computeFitness(current_scenario_std_for_individual, conf, scaler)
				
				predAction_list_current_gen.append(action) # Sarà sempre 0
				fitness_list_current_gen.append(individual_p.fitness)
				knob_list_current_gen.append(individual_p.knob) # Output grezzo della NN
			else:
				print(f"Warning: Mismatch between population size and number of standardized scenarios at index {i}")


		evaluation_end_time = time.time()
		print(f"Done > Evaluation took {evaluation_end_time - evaluation_start_time:.2f} sec")
		
		# Calcola il 'knob' medio della generazione (output medio della NN)
		gen_avg_knob_score = np.average(knob_list_current_gen) if knob_list_current_gen else 0.0

		# >>>>>> Calcolo del Self-Esteem (Stigma/Onore) <<<<<<
		# Questo passo aggiorna la fitness di ogni individuo basandosi su come la sua azione
		# si confronta con l'azione "media" della comunità (derivata da gen_avg_knob_score).
		
		#for individual_p in population:
		#	individual_p.computeSelfEsteem(conf, gen_avg_knob_score) 
		
		# Aggiorna la lista delle fitness dopo il calcolo del self-esteem
		fitness_list_current_gen = [p.fitness for p in population]


		# >>>>>> Registrazione dei dati per la generazione corrente <<<<<<
		# Costruisci il DataFrame df_results_current_gen riga per riga
		temp_rows_for_df = []
		for i, individual_p in enumerate(population):
			# Prendi lo scenario originale (denormalizzato) dalla df_scenari_input_for_nn
			if df_scenari_input_for_nn.shape[0] > i:
				original_scenario_data = df_scenari_input_for_nn.iloc[i]

				# Estrai i valori per il calcolo di 'convieneSvolta'
				s_numPed = original_scenario_data["numberOfPedestrians"]
				s_probPed = original_scenario_data["probPed"]
				s_numPass = original_scenario_data["numberOfPassengers"]
				s_probPass = original_scenario_data["probPass"]
				# L'altruismo per il calcolo di convieneSvolta è quello dello scenario/individuo
				s_altruism_scenario = original_scenario_data["AltruismLevel"] 
				s_selfish_scenario = 1 - s_altruism_scenario

				# Calcolo di 'convieneSvolta' (scelta utilitaristica ideale)
				# Questa logica determina se, da un punto di vista puramente utilitaristico
				# (secondo le formule originali del tuo codice), converrebbe svoltare.
				computeCostForSvoltaFlag = 0 # Flag per il costo aggiuntivo
				if (s_probPed * s_numPed > s_probPass * s_numPass): # Condizione originale
					computeCostForSvoltaFlag = 1

				# Utilità della SVOLTA (dal punto di vista utilitaristico)
				# Formula originale da ga_general.py (riga ~118):
				# u_svolta = scenario[0] * scenario[4] * scenario[1] - scenario[2] * (1 - scenario[4]) + scenario[0] * computeCost * conf.costPedestrian * scenario[1]
				# Adattata:
				util_svolta = (s_numPed * s_altruism_scenario * s_probPed) - \
							  (s_numPass * s_selfish_scenario) + \
							  (s_numPed * computeCostForSvoltaFlag * conf.costPedestrian * s_probPed)

				# Utilità dell'ANDARE DRITTO (dal punto di vista utilitaristico)
				# Formula originale da ga_general.py (riga ~119):
				# u_dritto = scenario[2] * (1-scenario[4]) * scenario[3] - scenario[0]*scenario[4] - scenario[0] * computeCost * conf.costPedestrian
				# Adattata (nota: l'originale non moltiplicava l'ultimo termine per s_probPed):
				util_dritto = (s_numPass * s_selfish_scenario * s_probPass) - \
							  (s_numPed * s_altruism_scenario) - \
							  (s_numPed * computeCostForSvoltaFlag * conf.costPedestrian) # L'originale non aveva s_probPed qui

				conviene_svolta_utilitarian_bool = (util_svolta > util_dritto)

				# Prepara la riga di dati da aggiungere al DataFrame
				row_data_dict = {
					"numberOfPedestrians": s_numPed,
					"probPed": s_probPed,
					"numberOfPassengers": s_numPass,
					"probPass": s_probPass,
					"AltruismLevel": s_altruism_scenario, # Altruismo usato per lo scenario
					"KnobLevel": knob_list_current_gen[i] if i < len(knob_list_current_gen) else -1, # Output NN
					"Fitness": fitness_list_current_gen[i] if i < len(fitness_list_current_gen) else -1, # Fitness (post self-esteem)
					'convieneSvolta': int(conviene_svolta_utilitarian_bool), # Scelta utilitaristica (0 o 1)
					"predAction": predAction_list_current_gen[i] if i < len(predAction_list_current_gen) else -1 # Azione dell'agente (sempre 0)
				}
				temp_rows_for_df.append(row_data_dict)
			else:
				print(f"Warning: Mismatch in data for individual {i} during logging.")


		df_results_current_gen = pd.DataFrame(temp_rows_for_df)
		# Opzionale: accumulare i risultati di tutte le generazioni
		# if not df_results_current_gen.empty:
		# 	all_generations_results_df = pd.concat([all_generations_results_df, df_results_current_gen], ignore_index=True)


		# Calcola le metriche della generazione
		gen_avg_fitness_score = np.average(fitness_list_current_gen) if fitness_list_current_gen else 0.0
		gen_min_fitness_score = np.min(fitness_list_current_gen) if fitness_list_current_gen else 0.0
		gen_max_fitness_score = np.max(fitness_list_current_gen) if fitness_list_current_gen else 0.0
		
		print(f"\tGeneration Metrics: Worst Fitness:{gen_min_fitness_score:.4f} | Avg Fitness:{gen_avg_fitness_score:.4f} | Best Fitness:{gen_max_fitness_score:.4f} | Avg Knob:{gen_avg_knob_score:.4f}")
		
		# Salva i risultati della generazione corrente
		if not df_results_current_gen.empty:
			save_results(df_results_current_gen, conf, current_generation) # Funzione da Individual.py
		
		# >>>>>> Selezione Genetica, Creazione Figli e Mutazione <<<<<<
		if conf.POPULATION_SIZE > 0 and population: # Solo se la popolazione non è vuota e la dimensione è > 0
			population_new = evolve_population(population, conf) # Funzione da Individual.py
		else:
			population_new = [] # Popolazione vuota se la dimensione della popolazione è 0 o la popolazione corrente è vuota


	# >>>>>> Calcolo Accuratezza Finale (sui dati dell'ultima generazione) <<<<<<
	print("\n********** Final Accuracy Calculation (based on last generation data) **********")
	
	# Usa i dati dell'ultima generazione (df_results_current_gen) per il calcolo dell'accuratezza.
	# Se df_results_current_gen è vuota (es. 0 generazioni), non fare nulla.
	if df_results_current_gen.empty:
		print("No data from the last generation to calculate accuracy.")
		if __name__ == "__main__": # Solo se eseguito come script principale
			simulation_end_time = time.time()
			print(f"\nTotal simulation time: {simulation_end_time - simulation_start_time:.2f} sec")
		return # Esce dalla funzione main se non ci sono dati

	# DataFrame per le statistiche di accuratezza aggregate
	df_accuracy_summary = pd.DataFrame(columns=['population_type','fp','fn','tp','tn']) 
	# DataFrame per le predizioni dettagliate (true_y vs pred_y)
	df_accuracy_detailed_predictions_rows = [] 

	# L'azione dell'agente deontologico è sempre "non svoltare" (cioè, predAction = 0).
	# Convertiamo in numerico per il confronto (0 per dritto/non svolta, 1 per svolta).
	agent_deontological_action_numeric = pd.Series([0] * len(df_results_current_gen)) # Sempre 0 (dritto)

	# 'convieneSvolta' dall'ultima generazione (già 0 o 1, rappresenta la scelta utilitaristica)
	utilitarian_ideal_choice_numeric = df_results_current_gen['convieneSvolta'].astype(int)

	# Calcolo di True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
	# per l'intera popolazione dell'ultima generazione, confrontando l'azione fissa dell'agente
	# con la scelta utilitaristica ideale per ogni scenario affrontato.

	# Azione agente = 0 (Dritto); Azione ideale = 1 (Svolta) -> False Negative (FN)
	fn_count = sum((agent_deontological_action_numeric == 0) & (utilitarian_ideal_choice_numeric == 1))
	
	# Azione agente = 0 (Dritto); Azione ideale = 0 (Dritto) -> True Negative (TN)
	tn_count = sum((agent_deontological_action_numeric == 0) & (utilitarian_ideal_choice_numeric == 0))

	# Dato che l'agente deontologico va SEMPRE dritto (azione = 0):
	# TP (agente svolta=1, ideale svolta=1) sarà sempre 0.
	tp_count = 0 
	# FP (agente svolta=1, ideale dritto=0) sarà sempre 0.
	fp_count = 0
	
	# Riga per il sommario dell'accuratezza della popolazione deontologica
	accuracy_summary_row = {
		'population_type': "deontological_fixed_action", 
		'fp': fp_count, 
		'fn': fn_count, 
		'tp': tp_count, 
		'tn': tn_count
	}
	df_accuracy_summary = pd.DataFrame([accuracy_summary_row])

	# Popola i dati per il DataFrame dettagliato delle predizioni
	true_y_list = utilitarian_ideal_choice_numeric.values.tolist()
	pred_y_list = agent_deontological_action_numeric.values.tolist() # L'azione dell'agente
		
	for true_value, pred_value in zip(true_y_list, pred_y_list):
		df_accuracy_detailed_predictions_rows.append({'true_y': true_value, 'pred_y': pred_value})
	
	df_accuracy_detailed_predictions = pd.DataFrame(df_accuracy_detailed_predictions_rows)
		
	# Salva i file di accuratezza
	if not df_accuracy_summary.empty:
		save_accuracy(df_accuracy_summary, conf, file_name=f"accuracy_summary_gen{conf.MAX_GENERATIONS-1}.txt")
	if not df_accuracy_detailed_predictions.empty:
		save_accuracy(df_accuracy_detailed_predictions, conf, file_name=f"accuracy_detailed_predictions_gen{conf.MAX_GENERATIONS-1}.txt")
	
	print("Accuracy calculation completed and results saved.")

	
if __name__ == "__main__":
	
	print(f"********** Simulation Starting **********")
	simulation_start_time = time.time() # Registra il tempo di inizio
	main() # Esegui la funzione principale
	simulation_end_time = time.time() # Registra il tempo di fine
	print(f"********** Simulation Finished **********")
	print(f"Total simulation duration: {simulation_end_time - simulation_start_time:.2f} seconds")