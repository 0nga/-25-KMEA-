# -*- coding: utf-8 -*-

import sys
import glob
import math
import os
import subprocess
import cPickle as pickle

from results import *

# altruism threshold
altruismThreshold=float(sys.argv[1])

# result folder
folder = sys.argv[2]
folders = sorted(glob.glob(folder + '/1*'), key=natural_key)

useLastGenOnly=True

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

# debug messages
print "Nr. repetitions", numRep
print "Nr. generations", numGen

# -------------------------------------------------------------------------------------------- #

# for each repetition
for i in range(numFolders):
    # if the repetition folder is not empty
    if not os.listdir(folders[i]) == []:

        print "Processing folder", folders[i]
        # get the generations files
        filename = folders[i] + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'
        dataFiles = sorted(glob.glob(filename), key=natural_key)
        
        if i == 0:
            command = "grep nrChambersPerRow -r " + folders[i] + "/out.txt | cut -f 2 -d ' '"
            numGroups = int(subprocess.check_output(['bash','-c', command]).splitlines()[0])
            
            # initialize the mean array (numRep x numGen x numGroups, numCol), containing
            # for each column and generation the column value averaged on all robots            
            if useLastGenOnly:
                meanGroupStats = np.empty((numRep,1,numGroups,4))
            else:
                meanGroupStats = np.empty((numRep,numGen,numGroups,4))
            meanGroupStats.fill(np.nan)
        
        # for each generation
        for gen in range(numGenInRep[i]):
            if useLastGenOnly:
                if gen < numGenInRep[i]-1:
                    continue
                j = 0
            else:
                j = gen
            
            # read the j-th generation file
            data = readData(dataFiles[gen])
            positionData = readPositionData(dataFiles[gen])

            '''
            # remove values of altruism level corresponding to 0 collected food
            nRobots = len(data)
            for k in range(nRobots):
                if data[k][collectedFoodIndex] == 0:
                    data[k][altruismLevelIndex] = np.nan
            '''
            
            # -------------------------------------------------------------------------------------------- #
            #                                        group statistics                                      #
            # -------------------------------------------------------------------------------------------- #
            groups = np.unique(positionData)
            k = 0
            for g in groups:
                fitnessInGroup = data[positionData==g,fitnessIndex]
                altruismInGroup = data[positionData==g,altruismLevelIndex]
                visitedChambersInGroup = data[positionData==g,chambersIndex]
                
                groupSize = len(data[positionData==g])
                fitnessMean = np.mean(fitnessInGroup)
                altruismMean = np.mean(altruismInGroup)
                nrDispIndividuals = len(visitedChambersInGroup[visitedChambersInGroup>1])
                
                meanGroupStats[i][j][k][0] = groupSize
                meanGroupStats[i][j][k][1] = fitnessMean
                meanGroupStats[i][j][k][2] = altruismMean
                meanGroupStats[i][j][k][3] = nrDispIndividuals
                k = k+1
                
# -------------------------------------------------------------------------------------------- #
#                                   serialize the data structures                              #
# -------------------------------------------------------------------------------------------- #
with open(folder + '/' + pickleFileGroupsStats, 'wb') as f:
    pickle.dump(meanGroupStats, f, protocol=2)
