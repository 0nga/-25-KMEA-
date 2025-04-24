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
tmpDataFile = folders[0] + '/' + 'logs'  + '/' + 'relatedness.txt'
numCol = np.size(readRelatednessData(tmpDataFile),1)

# generation array
gen = np.arange(0,numGen)

# debug messages
print "Nr. repetitions", numRep
print "Nr. generations", numGen

meanAggregate = np.empty((numRep,numGen,numCol))
meanAggregate.fill(np.nan)

# for each repetition
for i in range(numFolders):
    # if the repetition folder is not empty
    if not os.listdir(folders[i]) == []:

        print "Processing folder", folders[i]
        
        # get the relatedness files
        filename = folders[i] + '/' + 'logs'  + '/' + 'relatedness.txt'
        data = readRelatednessData(filename)
        
        # for each generation
        for j in range(numGenInRep[i]):
            # for each column
            for k in range(numCol):
                meanAggregate[i][j][k] = data[j][k]

# -------------------------------------------------------------------------------------------- #
#                                   serialize the data structures                              #
# -------------------------------------------------------------------------------------------- #
with open(folder + '/' + pickleFileRelatedness, 'wb') as f:
    pickle.dump(meanAggregate, f, protocol=2)