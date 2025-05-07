import os
import random
import pandas as pd
import numpy as np
import time
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, InputLayer, Input
from keras import backend as K
from Configuration import Configuration
import copy
from Individual import *
import sys, getopt

random.seed(42)

def compute_utilitarian_fitness(scenario, conf):
    """
    Compute fitness using a utilitarian approach:
    - Maximize total utility (sum of utilities for all affected parties)
    - No ethical knob, just pure utility maximization
    """
    numberOfPedestrians, probPed, numberOfPassengers, probPass = scenario[0:4]
    
    # Calculate expected utilities
    # For turning:
    # - Pedestrians: negative impact (probability of death * number of pedestrians)
    # - Passengers: positive impact (survival probability * number of passengers)
    utility_turn = (numberOfPassengers * (1 - probPass)) - (numberOfPedestrians * probPed)
    
    # For going straight:
    # - Pedestrians: positive impact (survival probability * number of pedestrians)
    # - Passengers: negative impact (probability of death * number of passengers)
    utility_straight = (numberOfPedestrians * (1 - probPed)) - (numberOfPassengers * probPass)
    
    # Return the action that maximizes total utility
    return 1 if utility_turn > utility_straight else 0

def main():
    # Create configuration object
    conf = Configuration()

    # Handle command line flags
    opts, args = getopt.getopt(sys.argv[1:], "ha:o:g:p:e:r")
    print(f"Arguments count: {len(sys.argv)}")
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("'test.py -g <numGenerations> -p <numIndividuals> -e <probDeath> -r'")
            sys.exit()
        elif opt in ("-g", "--gen"):
            conf.MAX_GENERATIONS = int(arg)
            print(f"MAX_GENERATIONS: {conf.MAX_GENERATIONS}")
        elif opt in ("-p",):
            conf.set_population_size(int(arg))
            print(f"POPULATION_SIZE: {conf.POPULATION_SIZE}")
        elif opt in ("-o",):
            conf.set_path(arg)
            print(f"path: {conf.path}")
        elif opt in ("-e",):
            conf.probDeathPedestrians = float(arg)
            print(f"probPed: {conf.probDeathPedestrians}")
        elif opt in ("-r",):
            conf.randomizeAltruism = True
            print(f"randomizeAltruism: {conf.randomizeAltruism}")

    # Save options
    save_options(conf)

    # Genetic Algorithm Section
    print("\n********** Genetic Algorithm (Utilitarian Approach) **********")

    # Generate initial population
    population_new = generate_first_population_randomly(conf)

    # Run Each Generation
    for current_generation in range(conf.MAX_GENERATIONS):
        population = population_new
        print(f"[+] Generation {current_generation+1} of {conf.MAX_GENERATIONS}")
        
        df = pd.DataFrame(columns=["numberOfPedestrians",
                                 "probPed",
                                 "numberOfPassengers",
                                 "probPass",
                                 "AltruismLevel",
                                 "Fitness",
                                 'convieneSvolta',
                                 "predAction"
                                 ])
        
        df_scenari = createScenarios(conf, population, conf.randomizeAltruism)
        scenari, scaler = standardize(df_scenari)

        # Evaluation Phase
        print(f"\tEvaluating Population: ", end='', flush=True)
        evaluation_start = time.time()

        predAction_list = []
        ind = 0

        for p in population:
            scenario = np.array(scenari[ind])
            scenario = scenario.reshape((1, 5))
            # Use utilitarian fitness function
            predAction = compute_utilitarian_fitness(scenario[0], conf)
            p.fitness = predAction  # Store the decision as fitness
            predAction_list.append(predAction)
            ind += 1

        evaluation_stop = time.time()
        print(f"Done > Takes {evaluation_stop - evaluation_start} sec")

        # Compute metrics for each individual
        ind = 0
        for p in population:
            scenario = np.array(df_scenari.iloc[ind])
            scenario = np.append(scenario, p.fitness)

            # Calculate if turning is better using utilitarian approach
            convieneSvolta = compute_utilitarian_fitness(scenario, conf)
            scenario = np.append(scenario, convieneSvolta)
            scenario = np.append(scenario, predAction_list[ind])

            df_temp = pd.DataFrame([scenario], columns=["numberOfPedestrians",
                                                      "probPed",
                                                      "numberOfPassengers",
                                                      "probPass",
                                                      "AltruismLevel",
                                                      "Fitness",
                                                      'convieneSvolta',
                                                      "predAction"
                                                      ])

            df = pd.concat([df, df_temp], ignore_index=True)
            ind += 1

        # Compute Generation Metrics
        gen_score_avg = np.average([item.fitness for item in population])
        gen_score_min = np.min([item.fitness for item in population])
        gen_score_max = np.max([item.fitness for item in population])

        print(f"\tWorst Score:{gen_score_min:.4f} | Average Score:{gen_score_avg:.4f} | Best Score:{gen_score_max:.4f}")

        save_results(df, conf, current_generation)

        # Genetic Selection, Children Creation and Mutation
        population_new = evolve_population(population, conf)

    # Save final accuracy metrics
    accuracyList = pd.DataFrame(columns=['individual', 'fp', 'fn', 'tp', 'tn'])
    accuracyList_pred = pd.DataFrame(columns=['true_y', 'pred_y'])

    for p in population:
        # Compute predictions using utilitarian approach
        scenarios = df_scenari.iloc[:, :4].values  # Only use the first 4 columns
        predictions = [compute_utilitarian_fitness(scenario, conf) for scenario in scenarios]
        
        # Calculate accuracy metrics
        tp = sum((df.convieneSvolta == True) & (np.array(predictions) == True))
        tn = sum((df.convieneSvolta == False) & (np.array(predictions) == False))
        fp = sum((df.convieneSvolta == False) & (np.array(predictions) == True))
        fn = sum((df.convieneSvolta == True) & (np.array(predictions) == False))

        accuracyList = pd.concat([accuracyList, pd.DataFrame([{'individual': p, 'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn}])], ignore_index=True)
        temp = pd.DataFrame(list(zip(df.convieneSvolta.values, predictions)), columns=['true_y', 'pred_y'])
        accuracyList_pred = pd.concat([accuracyList_pred, temp], ignore_index=True)

    save_accuracy(accuracyList, conf, None, conf.MAX_GENERATIONS)
    save_accuracy(accuracyList_pred, conf, "detailed_accuracy.txt")

if __name__ == "__main__":
    print(f"\tStart simulation: ", end='', flush=True)
    simulation_start = time.time()
    main()
    simulation_stop = time.time()
    print(f"Done > Takes {simulation_stop - simulation_start} sec") 