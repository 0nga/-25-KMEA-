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
for n in range(numFolders):
    if not os.listdir(folders[n]) == []:
        numRep = numRep+1

# number of generations for each repetition
numGenInRep = [0]*numRep
for n in range(numFolders):
    if not os.listdir(folders[n]) == []:
        dataFiles = sorted(glob.glob(folders[n] + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
        numGenInRep[n] = len(dataFiles)
numGen = np.max(numGenInRep)

# number of columns
tmpDataFiles = sorted(glob.glob(folders[0] + '/' + 'logs'  + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
numLoci = len(readGenomeData(tmpDataFiles[0])[0].split())

# debug messages
print "Nr. repetitions", numRep
print "Nr. generations", numGen

# numRep x numGen x numLoci x {min, mean, max}
if useRealValues:
    meanGenomes = np.empty((numRep,numGen,numLoci,3))
else:
    meanGenomes = np.empty((numRep,numGen,numLoci*nBits,3))
meanGenomes.fill(np.nan)

# for each repetition
for n in range(numFolders):
    # if the repetition folder is not empty
    if not os.listdir(folders[n]) == []:

        print "Processing folder", folders[n]
        # get the generations files
        filename = folders[n] + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'
        dataFiles = sorted(glob.glob(filename), key=natural_key)
        
        # for each generation
        for j in range(numGenInRep[n]):
            
            # read the j-th generation file
            genomeData = readGenomeData(dataFiles[j])
            nrRobots = len(genomeData)
        
            # for each robot
            for i in range(nrRobots):
                genome_i = genomeData[i].split()
                
                if i == 0 and j == 0:
                    # initialize the genome structure
                    if useRealValues:
                        genomes = np.zeros([numLoci,nrRobots])
                    else:
                        if genome_i[0][0] == '+' or genome_i[0][0] == '-':
                            print 'Error: the stored genome is real-valued'
                            sys.exit()
                        else:
                            genomes = np.zeros([numLoci*len(genome_i[0]),nrRobots])
                            
                # extract the genome of the i-th robot, in decimal format
                for k in range(numLoci):
                    if useRealValues:
                        if genome_i[k][0] == '+' or genome_i[k][0] == '-':
                            d = float(genome_i[k])
                        else:
                            d = scaleFromBits(int(genome_i[k]),nBits,geneMin,geneMax)
                        genomes[k][i] = d
                    else:
                        for b in range(nBits):
                            genomes[k*nBits+b][i] = float(genome_i[k][b])
                
            for k in range(numLoci):
                if useRealValues:
                    meanGenomes[n][j][k][0] = np.min(genomes[k])
                    meanGenomes[n][j][k][1] = np.mean(genomes[k])
                    meanGenomes[n][j][k][2] = np.max(genomes[k])
                else:
                    for b in range(nBits):
                        meanGenomes[n][j][k*nBits+b][0] = np.min(genomes[k*nBits+b])
                        meanGenomes[n][j][k*nBits+b][1] = np.mean(genomes[k*nBits+b])
                        meanGenomes[n][j][k*nBits+b][2] = np.max(genomes[k*nBits+b])

# -------------------------------------------------------------------------------------------- #
#                                   serialize the data structures                              #
# -------------------------------------------------------------------------------------------- #
with open(folder + '/' + pickleFileGenomes, 'wb') as f:
    pickle.dump(meanGenomes, f, protocol=2)