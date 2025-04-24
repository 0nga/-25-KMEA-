# -*- coding: utf-8 -*-

import sys
import glob
import math
import os
import cPickle as pickle

from results import *

# altruism threshold
altruismThreshold=float(sys.argv[1])

# result folder
folder = sys.argv[2]
folders = sorted(glob.glob(folder + '/1*'), key=natural_key)

# -------------------------------------------------------------------------------------------- #

# number of repetition folders (some of them might be empty)
numFolders = len(folders)

# number of repetitions (non-empty folders)
numRep = 0
for i in range(numFolders):
    if not os.listdir(folders[i]) == []:
        numRep = numRep+1

# number of generations for each repetition
numGenInRep = [0]*numRep
for i in range(numFolders):
    if not os.listdir(folders[i]) == []:
        dataFiles = sorted(glob.glob(folders[i] + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
        numGenInRep[i] = len(dataFiles)
numGen = np.max(numGenInRep)

# number of columns
tmpDataFiles = sorted(glob.glob(folders[0] + '/' + 'logs'  + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
numCol = np.size(readData(tmpDataFiles[0]),1)

# number of categories
numCategories = len(categories)

# generation array
gen = np.arange(0,numGen)

# debug messages
print "Nr. repetitions", numRep
print "Nr. generations", numGen

# initialize the mean array (numRep x numGen x numCol), containing
# for each column and generation the column value averaged on all robots
meanCategories = {}
for category in categories:
    meanCategories[category] = np.empty((numRep,numGen,numCol))
    meanCategories[category].fill(np.nan)

meanNrRobots = np.empty((numRep,numGen,numCategories))
meanNrRobots.fill(np.nan)

meanAggregate = np.empty((numRep,numGen,numCol))
meanAggregate.fill(np.nan)

meanForaging = np.empty((numRep,numGen))
meanForaging.fill(np.nan)

meanDispProb = np.empty((numRep,numGen))
meanDispProb.fill(np.nan)

meanNrGroups = np.empty((numRep,numGen,6))
meanNrGroups.fill(np.nan)

# for each repetition
for i in range(numFolders):
    # if the repetition folder is not empty
    if not os.listdir(folders[i]) == []:

        print "Processing folder", folders[i]
        # get the generations files
        filename = folders[i] + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'
        dataFiles = sorted(glob.glob(filename), key=natural_key)
        
        # for each generation
        for j in range(numGenInRep[i]):
            
            # read the j-th generation file
            data = readData(dataFiles[j])
            positionData = readPositionData(dataFiles[j])
            
            # remove values of altruism level corresponding to 0 collected food
            nRobots = len(data)
            #for k in range(nRobots):
            #    if data[k][collectedFoodIndex] == 0:
            #        data[k][altruismLevelIndex] = np.nan
        
            # -------------------------------------------------------------------------------------------- #
            #                      average the column value over each robot category                       #
            # -------------------------------------------------------------------------------------------- #
        
            # initialize the categories of robots for the i-th repetition (NOTE: their size is unknown)
            dataCategories = {}
            for category in categories:
                dataCategories[category] = []
            
            # discriminate here the various categories
            for k in range(nRobots):
                if True: #data[k][collectedFoodIndex] > 0:
                    # discriminate between selfish and altruist (based on altruism level)
                    if data[k][altruismLevelIndex] <= altruismThreshold:
                        if data[k][chambersIndex] <= 1:
                            dataCategories['static-selfish'] += [data[k]]
                        else:
                            dataCategories['dispersing-selfish'] += [data[k]]
                    else:
                        if data[k][chambersIndex] <= 1:
                            dataCategories['static-altruist'] += [data[k]]
                        else:
                            dataCategories['dispersing-altruist'] += [data[k]]
                else:
                    if 'static-non-forager' in categories:
                        if data[k][chambersIndex] <= 1:
                            dataCategories['static-non-forager'] += [data[k]]
                        else:
                            dataCategories['dispersing-non-forager'] += [data[k]]
                                
            # convert to numpy arrays
            for category in categories:
                dataCategories[category] = np.array(dataCategories[category])

            # separate measure over each category
            meanCategoriesInRep = {}
            for category in categories:
                if len(dataCategories[category]) > 0:
                    dataCategories[category] = np.ma.masked_array(dataCategories[category],np.isnan(dataCategories[category]))
                    meanCategoriesInRep[category] = np.mean(dataCategories[category],axis=0)
                    meanCategoriesInRep[category] = meanCategoriesInRep[category].filled(np.nan)
                else:
                    meanCategoriesInRep[category] = np.empty(numCol)
                    meanCategoriesInRep[category].fill(np.nan)

            for k in range(numCol):
                for category in categories:
                    meanCategories[category][i][j][k] = meanCategoriesInRep[category][k]

            # check if the sum of the category lengths equals the tot. nr. of robots
            sumCategories = 0
            for category in categories:
                sumCategories += len(dataCategories[category])
            if sumCategories < nRobots:
                print 'Error!'

            # -------------------------------------------------------------------------------------------- #
            #                               number of robots in each category                              #
            # -------------------------------------------------------------------------------------------- #
            for k in range(numCategories):
                meanNrRobots[i][j][k] = len(dataCategories[categories[k]])
        
            # -------------------------------------------------------------------------------------------- #
            #                           average the column value over all robots                           #
            # -------------------------------------------------------------------------------------------- #

            # aggregate measure over the entire population (no categories)
            data = np.ma.masked_array(data,np.isnan(data))
            mean = np.mean(data,axis=0)
            mean = mean.filled(np.nan)
            
            for k in range(numCol):
                meanAggregate[i][j][k] = mean[k]

            # -------------------------------------------------------------------------------------------- #
            #                                           foraging                                           #
            # -------------------------------------------------------------------------------------------- #
            #collectedFood = data[:,collectedFoodIndex]
            #meanForaging[i][j] = np.sum(collectedFood)

            # -------------------------------------------------------------------------------------------- #
            #                                     dispersal probability                                    #
            # -------------------------------------------------------------------------------------------- #
            visitedChambers = data[:,chambersIndex]
            meanDispProb[i][j] = float(len(visitedChambers[[visitedChambers>1]]))/len(visitedChambers)

            # -------------------------------------------------------------------------------------------- #
            #                                        group statistics                                      #
            # -------------------------------------------------------------------------------------------- #
            groups = np.unique(positionData)
            fitnessMean = []
            fitnessStdDev = []
            groupSize = []
            for g in groups:
                fitnessInGroup = data[positionData==g,fitnessIndex]
                fitnessMean += [np.mean(fitnessInGroup)]
                fitnessStdDev += [np.std(fitnessInGroup)]
                groupSize += [len(data[positionData==g])]

            meanNrGroups[i][j][0] = len(groups)             # nr or groups
            meanNrGroups[i][j][1] = np.std(fitnessMean)     # inter-group avg. fitness variance
            meanNrGroups[i][j][2] = np.mean(fitnessStdDev)  # avg. intra-group fitness variance
            meanNrGroups[i][j][3] = np.mean(groupSize)      # avg. group size
            meanNrGroups[i][j][4] = np.min(groupSize)       # min group size
            meanNrGroups[i][j][5] = np.max(groupSize)       # max group size

# -------------------------------------------------------------------------------------------- #
#                                   serialize the data structures                              #
# -------------------------------------------------------------------------------------------- #
for category in categories:
    with open(folder + '/' + pickleFileCategories[category], 'wb') as f:
        pickle.dump(meanCategories[category], f, protocol=2)

with open(folder + '/' + pickleFileAggregate, 'wb') as f:
    pickle.dump(meanAggregate, f, protocol=2)

with open(folder + '/' + pickleFileNrRobots, 'wb') as f:
    pickle.dump(meanNrRobots, f, protocol=2)

with open(folder + '/' + pickleFileForaging, 'wb') as f:
    pickle.dump(meanForaging, f, protocol=2)

with open(folder + '/' + pickleFileDispProb, 'wb') as f:
    pickle.dump(meanDispProb, f, protocol=2)

with open(folder + '/' + pickleFileNrGroups, 'wb') as f:
    pickle.dump(meanNrGroups, f, protocol=2)
