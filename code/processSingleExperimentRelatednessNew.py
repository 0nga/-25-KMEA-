# -*- coding: utf-8 -*-

import sys
import glob
import math
import os
import cPickle as pickle

from results import *

# altruism threshold
maxCol=int(sys.argv[1])

# relatednessFile
relatednessFile=sys.argv[2]

# result folder
folder = sys.argv[3]
folders = sorted(glob.glob(folder + '/1*'), key=natural_key)

# -------------------------------------------------------------------------------------------- #

def readRelatednessDataNew(filename, maxCol):
    data = np.loadtxt(filename,delimiter="\t",usecols=range(0,maxCol))
    return data

# -------------------------------------------------------------------------------------------- #

# number of repetition folders (some of them might be empty)
numFolders = len(folders)

# number of repetitions (non-empty folders)
numRep = 0
for i in range(numFolders):
    if not os.listdir(folders[i]) == []:
        numRep = numRep+1

tmpDataFile = folders[0] + '/' + 'logs'  + '/' + relatednessFile + '.txt'
tmpData = readRelatednessDataNew(tmpDataFile, maxCol)
# number of columns
numCol = np.size(tmpData,1)
# number of generations for each repetition
numGen = np.size(tmpData,0)

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
        filename = folders[i] + '/' + 'logs'  + '/' + relatednessFile + '.txt'
        data = readRelatednessDataNew(filename, maxCol)
        numGenInRep = np.size(data,0) 
        # for each generation
        for j in range(numGenInRep):
            # for each column
            for k in range(numCol):
                meanAggregate[i][j][k] = data[j][k]

# -------------------------------------------------------------------------------------------- #
#                                   serialize the data structures                              #
# -------------------------------------------------------------------------------------------- #
pickleFileRelatedness = 'results_' + relatednessFile + '.pickle'

with open(folder + '/' + pickleFileRelatedness, 'wb') as f:
    pickle.dump(meanAggregate, f, protocol=2)