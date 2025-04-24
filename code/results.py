# -*- coding: utf-8 -*-

import numpy as np

import re

# -------------------------------------------------------------------------------------------- #
#                               data column names and indexes                                  #
# -------------------------------------------------------------------------------------------- #

'''
# column indexes for full output 
columns = ['Shared food',
           'Collected food',
           'Altruism level',      # (#Shared food/#Collected food)
           'Altruism time',       # (Altruistic time/Total life time)
           'Dispersal distance',  # (#Chambers away from native chamber)
           'Chambers visited',
           'Patches visited',
           'Fitness',
           'Inclusive fitness',
           'Shared food (native)',
           'Collected food (native)',
           'Altruism time (native)']

sharedFoodIndex     = 0
collectedFoodIndex  = 1
altruismLevelIndex  = 2
altruismTimeIndex   = 3
dispersalIndex      = 4
chambersIndex       = 5
patchesIndex        = 6
fitnessIndex        = 7
inclusiveFitIndex   = 8

sharedFoodIndexNative     = 9
collectedFoodIndexNative  = 10
altruismTimeIndexNative   = 11

positionIndex       = 12
birthChamberIndex   = 13

idIndex             = 14
parentsIndex        = 15
genomeIndex         = 16

# column ranges for columns 0-12
colRanges = [[0,0],
             [0,0],
             [0,1],
             [0,1],
             [0,1],
             [0,0],
             [0,0],
             [0,0],
             [0,0],
             [0,0],
             [0,0],
             [0,1]]
'''

# column indexes for reduced output 
columns = ['Altruism level',      # (#Shared food/#Collected food)
           'Fitness',
           ]

altruismLevelIndex  = 0
fitnessIndex        = 1
genomeIndex         = 2

colRanges = [[0,10],
             [0,10]]

# min/max gene values (0,1 for direct encoding)
geneMin = -10
geneMax = 10
grayEncoding = True         # use Gray encoding
nBits = 8                   # number of bits per locus
useRealValues=True          # use real values for genotype analysis

# max number of active figures
maxFigures = 55

# -------------------------------------------------------------------------------------------- #
#                                   categories of behavior                                     #
# -------------------------------------------------------------------------------------------- #

'''
# this is the case with foraging (food patches)

# category names as they appear in the suffix of the pickle file names
categories = ['static-selfish', 'dispersing-selfish',
              'static-altruist', 'dispersing-altruist',
              'static-non-forager', 'dispersing-non-forager']

# category names as they appear in the plots
categoryNames = ['Non-dispersing selfish', 'Dispersing selfish',
                 'Non-dispersing altruist', 'Dispersing altruist',
                 'Non-dispersing unsuccessful', 'Dispersing unsuccessful']
                 
categoryShort = ['NS', 'DS',
                 'NA', 'DA',
                 'NU', 'DU']
'''

# this is the case with no foraging (public goods)

# category names as they appear in the suffix of the pickle file names
categories = ['static-selfish', 'dispersing-selfish',
              'static-altruist', 'dispersing-altruist']

# category names as they appear in the plots
categoryNames = ['Non-dispersing selfish', 'Dispersing selfish',
                 'Non-dispersing altruist', 'Dispersing altruist']

categoryShort = ['NS', 'DS',
                 'NA', 'DA']

'''
# this is the case with no altruism

# category names as they appear in the suffix of the pickle file names
categories = ['static-selfish', 'dispersing-selfish',
              'static-non-forager', 'dispersing-non-forager']

# category names as they appear in the plots
categoryNames = ['Non-dispersing successful', 'Dispersing successful',
                 'Non-dispersing unsuccessful', 'Dispersing unsuccessful']
                 
categoryShort = ['NS', 'DS',
                 'NU', 'DU']
'''

# categories files
pickleFileCategories = {}
for category in categories:
    pickleFileCategories[category] = 'results' + '_' + category + '.pickle'
# number of robots in each category
pickleFileNrRobots = 'results_robots.pickle'
# aggregate file
pickleFileAggregate = 'results.pickle'
# foraging
pickleFileForaging = 'results_foraging.pickle'
# dispersal probability
pickleFileDispProb = 'results_disp_prob.pickle'
# general group statistics (nr or groups, inter-group avg. fitness variance, avg. intra-group fitness variance, avg./min/max group size)
pickleFileNrGroups = 'results_nr_groups.pickle'
# detailed group statistics (group size, avg. fitness in group, avg. altruism in group, nr. of migrants)
pickleFileGroupsStats = 'results_groups.pickle'
# relatedness
pickleFileRelatedness = 'results_relatedness.pickle'
# genomes
pickleFileGenomes = 'results_genomes.pickle'

# color map
colorMap = 'RdYlGn'

'''
#Other options:
'Set1'
'RdYlBu'
'autumn'
'bone'
'BrBG'
'brg'
'cool'
'coolwarm'
'gist_earth'
'gray'
'ocean'
'PiYG'
'PRGn'
'RdBu'
'RdYlBu'
'summer'
'Wistia'
'YlGnBu'
'''

# -------------------------------------------------------------------------------------------- #
#                                       utility functions                                      #
# -------------------------------------------------------------------------------------------- #
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def readData(filename):
    data = np.loadtxt(filename,delimiter="\t",skiprows=1,usecols=range(0,len(colRanges)))
    return data

def readGenomeData(filename):
    data = np.loadtxt(filename,dtype='string',delimiter="\t",skiprows=1,usecols=(genomeIndex,))
    return data

# -------------------------------------------------------------------------------------------- #
def getNrGroups(filename):
    search_term = 'nrChambersPerRow'
    for line in open(filename, 'r'):
        if re.search(search_term, line):
            return int(line.split(': ')[1])

# -------------------------------------------------------------------------------------------- #
def scaleToBits(x, nBits, minX, maxX):
    half = (1 << nBits-1)-1
    mask = (1 << nBits)-1
    
    range = float(maxX-minX)
    eps = float(range)/(mask+1)
    
    if (maxX == -minX):
        # symmetric ranges
        if (x < -range/(mask-1) + eps):
            w = int(((x-minX)*mask/range)) & mask
        elif (x > range/(mask-1) - eps):
            w = (int((x-minX)*(mask-1)/range) & mask)+1
        else:
            w = half
    else:
        # asymmetric ranges
        w = (int((x-minX)*(mask+1)/range) & mask)
        if (x == maxX):
            w = mask
    
    if grayEncoding:
        w = binaryToGray(w,nBits)
    return w
        
def scaleFromBits(x, nBits, minX, maxX):
    if grayEncoding:
        x = grayToBinary(x,nBits)
    
    half = (1 << nBits-1)-1
    mask = (1 << nBits)-1
    
    d = float(x & mask)
    
    if (maxX == -minX):
        # symmetric ranges
        if (d < half):
            return d*float(-minX)/half + minX
        elif (d > half+1):
            return (d-1)*float(maxX)/half + minX
        else:
            return 0
    else:
        # asymmetric ranges
        return d*float(maxX-minX)/(half*2+1) + minX

def binaryToGray(x, nBits):
    x_ = x & ((1 << nBits)-1)
    return (x_ >> 1) ^ x_
    
def grayToBinary(x, nBits):
    mask = (1 << nBits)-1
    x_ = x & mask
    mask = x_ >> 1
    while mask != 0:
        x_ = x_ ^ mask
        mask = mask >> 1
    return x_

def toBitString(x, nBits):
    mask = (1 << nBits)-1
    s = bin(x & mask)[2:]
    n = len(s)
    pad = ""
    i = 0
    while i < (nBits-n):
        pad += "0"
        i = i+1
    s = (pad + s) if (n < nBits) else s
    return s

def test():
    minX = -6
    maxX = 6
    
    nBits = 8
    bitMask = (1 << nBits)-1
    scaled = 0
    
    grayEncoding = False
    while (scaled <= bitMask):
        d = scaleFromBits(scaled, nBits, minX, maxX)
        scaledTest = scaleToBits(d, nBits, minX, maxX)
        
        print( toBitString(scaled, nBits) + "\t" + \
                toBitString(scaledTest, nBits) + "\t" + \
                str(((scaled & bitMask)-(scaledTest & bitMask))) + "\t" + str(d))
        scaled = scaled+1
    
    print ("***************")
    
    nBits = 8
    bitMask = (1 << nBits)-1
    scaled = 0
    
    grayEncoding = True
    while (scaled <= bitMask):
        gray = binaryToGray(scaled, nBits)
        binary = grayToBinary(gray, nBits)
        
        print( toBitString(scaled, nBits) + "\t" + \
                toBitString(gray, nBits) + "\t" + \
                toBitString(binary, nBits) + "\t" + \
                str(((scaled & bitMask)-(binary & bitMask))))
        scaled = scaled+1
