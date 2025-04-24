# -*- coding: utf-8 -*-

import sys
import glob
import math
import os
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')

from scipy.stats import *
from pylab import *

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

from results import *

# disable interactive mode (to create figures in background)
ioff()

# legend properties
legendSize = 14

# figures format
figureFormat='.pdf' #.svg .png .eps .pdf

# compute standard deviation
computeStd = True

# choose what to compare (to create the plot legends)
# possible values: {"D", "Y", "L", "SelectionTypes", "DensityInputs", "Inputs", "ZeroInputs", "AltrNonAltr", "n", "t"}
compare = sys.argv[1]

# result folder
folders = sys.argv[2:]

# max generation
#maxGen = 200
maxGen = np.inf

# category styles for plots of number of individuals per category
style = ['r:', 'r', 'g:', 'g', 'b:', 'b']
width = [2, 2, 2, 2, 2, 2]

# category styles for correlation plots
style_corr = ['ro', 'rx', 'go', 'gx', 'bo', 'bx']
width_corr = [2, 2, 2, 2, 2, 2]

# number of treatments
numTreatments=len(folders)

# legends (one per treatment)
legends = ['']*numTreatments

# number of categories
numCategories=len(categories)

# color map
NUM_COLORS = numTreatments
cm = plt.get_cmap(colorMap)
values=range(NUM_COLORS)
cNorm = colors.Normalize(vmin=values[0], vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

#mycolors = [cm(1.*n/NUM_COLORS) for n in values]    # manual scaling
mycolors = [scalarMap.to_rgba(n) for n in values]   # map scaling

#print mpl.rcParams
mpl.rcParams['figure.max_open_warning']=maxFigures

# number or rows/columns in subplots
if numTreatments == 1:
    numRowFig=1
    numColFig=1
else:
    if numTreatments % 2 == 0:
        numRowFig=2
        numColFig=numTreatments/2
    elif numTreatments % 3 == 0:
        numRowFig=3
        numColFig=numTreatments/3
    else:
        numRowFig=1
        numColFig=numTreatments

# incremental index
n = 0

# -------------------------------------------------------------------------------------------- #

# for each experiment configuration
for folder in folders:
    print "Processing folder", folder
    
    # -------------------------------------------------------------------------------------------- #
    # create legend
    if numTreatments > 1:
        if folder.endswith('/'):
            folder=folder[:-1]
        options=folder.split('-')[1:]
        if compare == "D":
            for option in options:
                if option.startswith('D'):
                    legends[n] = 'D='+option[1:]
        elif compare == "Y":
            for option in options:
                if option.startswith('Y'):
                    legends[n] = 'Y='+option[1:]
        elif compare == "L":
            for option in options:
                if option.startswith('L'):
                    legends[n] = 'L='+option[1:]
        elif compare == "i":
            for option in options:
                if option.startswith('i'):
                    legends[n] = 'i='+option[1:]
        elif compare == "SelectionTypes":
            for option in options:
                if option.startswith('c'):
                    legends[n] = option[1:]
        elif compare == "DensityInputs":
            if 'd' in options and 'f' in options:
                legends[n] = 'pop,food'
            elif 'd' in options:
                legends[n] = 'pop'
            elif 'f' in options:
                legends[n] = 'food'
            else:
                legends[n] = 'none'
        elif compare == "Inputs":
            if 'd' in options and 'f' in options:
                if 'e' in options:
                    legends[n] = 'pop,food,distance,chamber'
                else:
                    legends[n] = 'pop,food,chamber'
            else:
                if 'm' in options:
                    legends[n] = 'chamber'
                else:
                    legends[n] = 'none'
        elif compare == "ZeroInputs":
            if n == 0:
                legends[n] = 'pop,dist'
            elif n == 1:
                legends[n] = 'pop'
            elif n == 2:
                legends[n] = 'dist'
            elif n == 3:
                legends[n] = 'none'
        elif compare == "AltNonAltr":
            if 'a' in options:
                legends[n] = 'no-altruism'
            else:
                if 'Y' in options:
                    for option in options:
                        if option.startswith('Y'):
                            Ylegend = '(Y='+option[1:]+')'
                else:
                    Ylegend = '(Y=10)'
                legends[n] = 'altruism'+' '+Ylegend
        elif compare == "n":
            for option in options:
                if option.startswith('n'):
                    legends[n] = 'n='+option[1:]
        elif compare == "t":
            for option in options:
                if option.startswith('t'):
                    legends[n] = 't='+option[1:]
                      
    # -------------------------------------------------------------------------------------------- #
    # plot group stats (group size, avg. fitness in group, avg. altruism in group, nr. of migrants)
    print "Processing pickle file", pickleFileGroupsStats
    with open(folder + '/' + pickleFileGroupsStats, 'rb') as f:
        meanGroupStats = pickle.load(f)

    if n == 0:
        numRep = len(meanGroupStats)
        #numGen = len(meanGroupStats[0])
        #lastGen = numGen-1
        lastGen = 0
        options=folder.split('-')[1:]
        for option in options:
            if option.startswith('nrChambersPerRow'):
                numGroups = int(option[len('nrChambersPerRow'):])        
        numColGroupStats = len(meanGroupStats[0][0][0]) # number of group stats
        
    groupStatsFigLabels = ['group_size','fitness_mean','altruism_mean','nr_migrants']
    groupStatsYLabels = ['Group size','Avg. fitness','Avg. altruism','% Migrants']

    colRanges = [[0,0],
                 [0,0],
                 [0,1],
                 [0,1]]

    figure('corr_groups_stats_'+legends[n], figsize=(18, 11.25)) 
    print 'Creating figure:', 'corr_groups_stats_'+legends[n]
    f = 0
    
    # consider the last generation only
    
    maxSize = 0
    maxFitness = 0
    for j in range(numRep):
        for k in range(numGroups):
            if meanGroupStats[j][lastGen][k][0] > maxSize:
                maxSize = meanGroupStats[j][lastGen][k][0]
            if meanGroupStats[j][lastGen][k][1] > maxFitness:
                maxFitness = meanGroupStats[j][lastGen][k][1]
    
    for gs1 in range(numColGroupStats):
        for gs2 in range(numColGroupStats):
            if gs1 > gs2:
                ax=subplot(numColGroupStats-1,numColGroupStats-1,f+1-numColGroupStats-gs1+1)
                
                ax.locator_params(axis='both',tight=False,nbins=4)
                
                xdata = []
                ydata = []
                for j in range(numRep):
                    for k in range(numGroups):
                        tmp1 = meanGroupStats[j][lastGen][k][gs2]
                        tmp2 = meanGroupStats[j][lastGen][k][gs1]
                        
                        #if gs2 == 0:
                        #    tmp1 = float(tmp1)/maxSize
                        ##elif gs2 == 1:
                        #    tmp1 = float(tmp1)/maxFitness
                        if gs2 == 3:
                            tmp1 = float(tmp1)/meanGroupStats[j][lastGen][k][0]
                        
                        #if gs1 == 0:
                        #    tmp2 = float(tmp2)/maxSize
                        #elif gs1 == 1:
                        #    tmp2 = float(tmp2)/maxFitness
                        if gs1 == 3:
                            tmp2 = float(tmp2)/meanGroupStats[j][lastGen][k][0]
                                            
                        xdata += [tmp1]
                        ydata += [tmp2]
                       
                plot(xdata, ydata, '.', color=mycolors[n])
                
                if (numColGroupStats*numColGroupStats-numColGroupStats) < (f+1) and (f+1) <= (numColGroupStats*numColGroupStats):
                    xlabel(groupStatsYLabels[gs2])
                else:
                    gca().xaxis.set_major_formatter(plt.NullFormatter())
                if f == 0 or (f % numColGroupStats) == 0:
                    ylabel(groupStatsYLabels[gs1])
                else:
                    gca().yaxis.set_major_formatter(plt.NullFormatter())
                
                if colRanges[gs2][0] != colRanges[gs2][1]:
                    xlim(colRanges[gs2])
                else:
                    minXdata = min(xdata)
                    maxXdata = max(xdata)
                    #print minXdata, maxXdata, maxXdata-minXdata < 1
                    if maxXdata-minXdata < 1:
                        gca().set_autoscale_on(False)
                        xlim([(minXdata+maxXdata)/2-0.5,(minXdata+maxXdata)/2+0.5])
                if colRanges[gs1][0] != colRanges[gs1][1]:
                    ylim(colRanges[gs1])
                else:
                    minYdata = min(ydata)
                    maxYdata = max(ydata)
                    #print minYdata, maxYdata, maxYdata-minYdata < 1
                    if maxYdata-minYdata < 1:
                        gca().set_autoscale_on(False)
                        ylim([(minYdata+maxYdata)/2-0.5,(minYdata+maxYdata)/2+0.5])
                savefig('corr_groups_stats_'+legends[n]+figureFormat)
        
            f = f+1

    n = n+1

# -------------------------------------------------------------------------------------------- #
#show()
