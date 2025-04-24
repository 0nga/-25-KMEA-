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
from sets import Set

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
# possible values: {"D", "Y", "q", "O", "L", "i", "n", "t"}
compare1 = sys.argv[1]
# possible values: {"D", "Y", "q", "O", "L", "i", "n", "t"}
compare2 = sys.argv[2]

# result folder
folders = sys.argv[3:]

# max generation
#maxGen = 20
maxGen = np.inf

# number of genes (including marker loci)
numGenes = 5

# used in rel-alt and rel-disp plots 
minVerticalHeight=0.05

# data indexes to plot
dataIndexes = [
               altruismLevelIndex,
               chambersIndex,
               fitnessIndex,
               ]

# number of plots
nPlots = len(dataIndexes)

# flier symbol in boxplot ('' for no fliers)
flierSymbol='b+'

groupStatsFigLabels = ['nr_groups','inter_group_diversity','intra_group_diversity','group_size_avg','group_size_min','group_size_max']
groupStatsYLabels = ['Number of groups','Inter-group fitness diversity','Intra-group fitness diversity','Group size avg', 'Group size min', 'Group size max']

# -------------------------------------------------------------------------------------------- #

# create legends
legend1 = Set()
legend2 = Set()

boxplotPositions = Set()

for folder in folders:
    if folder.endswith('/'):
        folder=folder[:-1]
    options=folder.split('-')[1:]
    for option in options:
        if compare1 == "D" and option.startswith('D'):
            legend1.add('D='+option[1:])
        elif compare1 == "Y" and option.startswith('Y'):
            legend1.add('Y='+option[1:])
        elif compare1 == "q" and option.startswith('q'):
            legend1.add('q='+option[1:])
        elif compare1 == "O" and option.startswith('O'):
            legend1.add('a='+option[1:])
        elif compare1 == "L" and option.startswith('L'):
            legend1.add('L='+option[1:])
        elif compare1 == "i" and option.startswith('i'):
            legend1.add('i='+option[1:])
        elif compare1 == "n" and option.startswith('n'):
            legend1.add('n='+option[1:])
        elif compare1 == "t" and option.startswith('t'):
            legend1.add('t='+option[1:])
        
        if compare2 == "D" and option.startswith('D'):
            legend2.add('D='+option[1:])
        elif compare2 == "Y" and option.startswith('Y'):
            legend2.add('Y='+option[1:])
            boxplotPositions.add(float(option[1:]))
        elif compare2 == "q" and option.startswith('q'):
            legend2.add('q='+option[1:])
        elif compare2 == "O" and option.startswith('O'):
            legend2.add('a='+option[1:])
        elif compare2 == "L" and option.startswith('L'):
            legend2.add('L='+option[1:])
        elif compare2 == "i" and option.startswith('i'):
            legend2.add('i='+option[1:])
        elif compare2 == "n" and option.startswith('n'):
            legend2.add('n='+option[1:])
        elif compare2 == "t" and option.startswith('t'):
            legend2.add('t='+option[1:])

legend1=list(legend1)
legend2=list(legend2)
legend1=sorted(legend1, key=lambda item: (float(item.partition('=')[1]) if item[1].isdigit() else float('inf'), item))
legend2=sorted(legend2, key=natural_key)

boxplotPositions=list(boxplotPositions)
boxplotPositions=np.array(boxplotPositions)
boxplotPositions=np.sort(boxplotPositions)

# number of experiments and treatments (per experiment)
numExperiments=len(legend1)
numTreatments=len(legend2)

# consider the last generations for calculating the boxplots
boxplotGenerations=1 #min(maxGen,100)

# color map
NUM_COLORS = numTreatments
cm = plt.get_cmap(colorMap)
values = range(NUM_COLORS)
cNorm = colors.Normalize(vmin=values[0], vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

#mycolors = [cm(1.*n/NUM_COLORS) for n in values]    # manual scaling
mycolors = [scalarMap.to_rgba(n) for n in values]   # map scaling

#print mpl.rcParams
mpl.rcParams['figure.max_open_warning']=maxFigures

# number or rows/columns in subplots
if numExperiments == 1:
    numRowFig=1
    numColFig=1
else:
    if numExperiments % 2 == 0:
        numRowFig=2
        numColFig=numExperiments/2
    elif numExperiments % 3 == 0:
        numRowFig=3
        numColFig=numExperiments/3
    else:
        numRowFig=1
        numColFig=numExperiments

firstFolder = True

# for each experiment configuration
for folder in folders:
    print "Processing folder", folder
    
    if folder.endswith('/'):
        folder=folder[:-1]
    options=folder.split('-')[1:]
    for option in options:
        if compare1 == "D" and option.startswith('D'):
            index1=legend1.index('D='+option[1:])
        elif compare1 == "Y" and option.startswith('Y'):
            index1=legend1.index('Y='+option[1:])
        elif compare1 == "q" and option.startswith('q'):
            index1=legend1.index('q='+option[1:])
        elif compare1 == "O" and option.startswith('O'):
            index1=legend1.index('a='+option[1:])
        elif compare1 == "L" and option.startswith('L'):
            index1=legend1.index('L='+option[1:])
        elif compare1 == "i" and option.startswith('i'):
            index1=legend1.index('i='+option[1:])
        elif compare1 == "n" and option.startswith('n'):
            index1=legend1.index('n='+option[1:])
        elif compare1 == "t" and option.startswith('t'):
            index1=legend1.index('t='+option[1:])
        
        if compare2 == "D" and option.startswith('D'):
            index2=legend2.index('D='+option[1:])
        elif compare2 == "Y" and option.startswith('Y'):
            index2=legend2.index('Y='+option[1:])
        elif compare2 == "q" and option.startswith('q'):
            index2=legend2.index('q='+option[1:])
        elif compare2 == "O" and option.startswith('O'):
            index2=legend2.index('a='+option[1:])
        elif compare2 == "L" and option.startswith('L'):
            index2=legend2.index('L='+option[1:])
        elif compare2 == "i" and option.startswith('i'):
            index2=legend2.index('i='+option[1:])
        elif compare2 == "n" and option.startswith('n'):
            index2=legend2.index('n='+option[1:])
        elif compare2 == "t" and option.startswith('t'):
            index2=legend2.index('T='+option[1:])
    
    # -------------------------------------------------------------------------------------------- #
    # load results from files
    print "Processing pickle file", pickleFileAggregate
    with open(folder + '/' + pickleFileAggregate, 'rb') as f:
        meanAggregate = pickle.load(f)
    
    # this file has 5 column: 5 rel
    print "Processing pickle file", pickleFileRelatedness
    with open(folder + '/' + pickleFileRelatedness, 'rb') as f:
        meanRelatedness = pickle.load(f)

    numRel = len(meanRelatedness[0][0])

    # this file has 20 columns: 5 rel + 3 params x 5 genes (alpha, -c, b)
    pickleFileRelatednessRegr1 = 'results_regression_1.pickle'
    print "Processing pickle file", pickleFileRelatednessRegr1
    with open(folder + '/' + pickleFileRelatednessRegr1, 'rb') as f:
        meanRelatednessRegression1 = pickle.load(f)

    numRelRegr1 = len(meanRelatednessRegression1[0][0])

    # this file has 11 columns: alpha + 5 non-social gradients + 5 social gradients
    pickleFileRelatednessRegrMultGen = 'results_regression_multi_gen.pickle'
    print "Processing pickle file", pickleFileRelatednessRegrMultGen
    with open(folder + '/' + pickleFileRelatednessRegrMultGen, 'rb') as f:
        meanRelatednessRegressionMultiPhen = pickle.load(f)

    numRelRegrMultPhen = len(meanRelatednessRegressionMultiPhen[0][0])

    if firstFolder:
        numRep = len(meanAggregate)
        numGen = min(len(meanAggregate[0]),maxGen)
        numCol = len(meanAggregate[0][0]) # number of robots' traits of interest

        # generation array
        gen = arange(0,numGen)
        
        if numRep == 1:
            dataFiles = sorted(glob.glob(folder + '/' + '1*' + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
            dataTmp = readData(dataFiles[0])
            
            boxPlotDataAggregate = np.empty((numExperiments,numCol,numTreatments,len(dataTmp)))
            boxPlotDataAggregate.fill(np.nan)
        else:
            # (numExperiments x numCol x numTreatments x numRep)
            boxPlotDataAggregate = np.empty((numExperiments,numCol,numTreatments,numRep))
            boxPlotDataAggregate.fill(np.nan)
        
         # mean and std. dev. (over numRep)
        mean = np.empty((numCol,numGen))
        if computeStd:
            std = np.empty((numCol,numGen))
        # single repetition data
        dataRep = np.empty(numRep)
        dataRep.fill(np.nan)
        
        minValues = np.empty(numCol)
        minValues.fill(np.inf)
        maxValues = np.empty(numCol)
        maxValues.fill(-np.inf)
        
        # (numExperiments x numRel x numTreatments x numRep)
        boxPlotDataRelatedness = np.empty((numExperiments,numRel,numTreatments,numRep))
        boxPlotDataRelatedness.fill(np.nan)
        
        # mean and std. dev. (over numRep)
        meanRel = np.empty((numRel,numGen))
        if computeStd:
            stdRel = np.empty((numRel,numGen))
        # single repetition data
        dataRepRel = np.empty(numRep)
        dataRepRel.fill(np.nan)
        
        minValuesRel = np.empty(numRel)
        minValuesRel.fill(np.inf)
        maxValuesRel = np.empty(numRel)
        maxValuesRel.fill(-np.inf)
        
        # (numExperiments x numRelRegr1 x numTreatments x numRep)
        boxPlotDataRelRegr1 = np.empty((numExperiments,numRelRegr1,numTreatments,numRep))
        boxPlotDataRelRegr1.fill(np.nan)
        
        # mean and std. dev. (over numRep)
        meanRelRegr1 = np.empty((numRelRegr1,numGen))
        if computeStd:
            stdRelRegr1 = np.empty((numRelRegr1,numGen))
        # single repetition data
        dataRepRelRegr1 = np.empty(numRep)
        dataRepRelRegr1.fill(np.nan)
        
        minValuesRelRegr1 = np.empty(numRelRegr1)
        minValuesRelRegr1.fill(np.inf)
        maxValuesRelRegr1 = np.empty(numRelRegr1)
        maxValuesRelRegr1.fill(-np.inf)
        
        # (numExperiments x numRelRegrMultPhen x numTreatments x numRep)
        boxPlotDataRelRegrMultPhen = np.empty((numExperiments,numRelRegrMultPhen,numTreatments,numRep))
        boxPlotDataRelRegrMultPhen.fill(np.nan)
        
        # mean and std. dev. (over numRep)
        meanRelRegrMultPhen = np.empty((numRelRegrMultPhen,numGen))
        if computeStd:
            stdRelRegrMultPhen = np.empty((numRelRegrMultPhen,numGen))
        # single repetition data
        dataRepRelRegrMultPhen = np.empty(numRep)
        dataRepRelRegrMultPhen.fill(np.nan)
        
        minValuesRelRegrMultPhen = np.empty(numRelRegrMultPhen)
        minValuesRelRegrMultPhen.fill(np.inf)
        maxValuesRelRegrMultPhen = np.empty(numRelRegrMultPhen)
        maxValuesRelRegrMultPhen.fill(-np.inf)
        
        # stats min/max values
        minValuesStats = np.empty(len(groupStatsFigLabels))
        minValuesStats.fill(np.inf)
        maxValuesStats = np.empty(len(groupStatsFigLabels))
        maxValuesStats.fill(-np.inf)

    if numRep == 1:
        dataFiles = sorted(glob.glob(folder + '/' + '1*' + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)

    dataLoaded = []
    for j in gen:
        dataLoaded += [False]
    
    data = []    
    
    # -------------------------------------------------------------------------------------------- #
    # plot the trends for each column along the generations
    for i in range(numCol):
        # take only some columns
        if i not in dataIndexes:
            continue
        
        '''
        figure(str(i)+'_aggregate', figsize=(18, 11.25))
        subplot(numRowFig,numColFig,index1+1)
        
        print 'Creating figure:', str(i)+'_aggregate'
        '''
        # for each generation
        for j in gen:
            if j < gen[-1]:
                continue
            
            if numRep == 1:
                if not dataLoaded[j]:
                    dataTmp = readData(dataFiles[j])
                    data += [dataTmp]
                    dataLoaded[j] = True
                
            # for each repetition
            for k in range(min(len(meanAggregate),numRep)):
                # collect the generation data of each repetition
                dataRep[k] = meanAggregate[k][j][i]
                
                if numRep == 1:
                    if j == gen[-1]:
                        boxPlotDataAggregate[index1][i][index2] = dataTmp[:,i]
                else:
                    # save the last generations data for boxplot
                    if j == gen[-1]-boxplotGenerations+1:
                        boxPlotDataAggregate[index1][i][index2][k] = dataRep[k]
                    elif j <= gen[-1]:
                        boxPlotDataAggregate[index1][i][index2][k] += dataRep[k]
                    if j == gen[-1]:
                        boxPlotDataAggregate[index1][i][index2][k] /= boxplotGenerations
            
            '''
            # for each generation, average among the repetitions
            if count_nonzero(~isnan(dataRep)):
                mean[i][j] = nanmean(dataRep)
                if computeStd:
                    if numRep == 1:
                        std[i][j] = nanstd(np.array(data[0])[:,i])
                    else:
                        std[i][j] = nanstd(dataRep)
            else:
                mean[i][j] = np.nan
                if computeStd:
                    if numRep == 1:
                        std[i][j] = nanstd(np.array(data[0])[:,i])
                    else:
                        std[i][j] = np.nan

        if computeStd:
            line = plot(gen, mean[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            fill_between(gen, mean[i]-std[i], mean[i]+std[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            tmpMin = min(mean[i]-std[i])
            tmpMax = max(mean[i]+std[i])
        else:
            plot(gen, mean[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            tmpMin = min(mean[i])
            tmpMax = max(mean[i])

        if tmpMin < minValues[i]:
            minValues[i] = tmpMin
        if tmpMax > maxValues[i]:
            maxValues[i] = tmpMax

        xlim([0,numGen])
        
        title(legend1[index1])
        if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
            xlabel('Generations')
        else:
            gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (index1+1) == 1 or (index1 % numColFig) == 0:
            ylabel(columns[i])
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
        
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                        numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
                
        savefig(str(i)+'_aggregate'+figureFormat)
    '''
    
    # -------------------------------------------------------------------------------------------- #
    # plot the trends for each column along the generations
    for i in range(numRel):
        
        '''        
        figure('rel'+str(i)+'_aggregate', figsize=(18, 11.25))
        subplot(numRowFig,numColFig,index1+1)
        
        print 'Creating figure:', 'rel'+str(i)+'_aggregate'
        '''
        # for each generation
        for j in gen:
            if j < gen[-1]:
                continue
            
            # for each repetition
            for k in range(min(len(meanRelatedness),numRep)):
                # collect the generation data of each repetition
                dataRepRel[k] = meanRelatedness[k][j][i]
                
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotDataRelatedness[index1][i][index2][k] = dataRepRel[k]
                elif j <= gen[-1]:
                    boxPlotDataRelatedness[index1][i][index2][k] += dataRepRel[k]
                if j == gen[-1]:
                    boxPlotDataRelatedness[index1][i][index2][k] /= boxplotGenerations
        
            '''
            # for each generation, average among the repetitions
            if count_nonzero(~isnan(dataRepRel)):
                meanRel[i][j] = nanmean(dataRepRel)
                if computeStd:
                    stdRel[i][j] = nanstd(dataRepRel)
            else:
                meanRel[i][j] = np.nan
                if computeStd:
                    stdRel[i][j] = np.nan

        if computeStd:
            line = plot(gen, meanRel[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            fill_between(gen, meanRel[i]-stdRel[i], meanRel[i]+stdRel[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            tmpMin = min(meanRel[i]-stdRel[i])
            tmpMax = max(meanRel[i]+stdRel[i])
        else:
            plot(gen, meanRel[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            tmpMin = min(meanRel[i])
            tmpMax = max(meanRel[i])

        if tmpMin < minValuesRel[i]:
            minValuesRel[i] = tmpMin
        if tmpMax > maxValuesRel[i]:
            maxValuesRel[i] = tmpMax

        xlim([0,numGen])
        
        title(legend1[index1])
        if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
            xlabel('Generations')
        else:
            gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (index1+1) == 1 or (index1 % numColFig) == 0:
            ylabel('Relatedness ' + '(' + str(i) + ')')
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
        
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                        numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
                
        savefig('rel'+str(i)+'_aggregate'+figureFormat)
    '''
    
    # -------------------------------------------------------------------------------------------- #
    # plot the trends for each column along the generations
    for i in range(numRelRegr1):
        '''    
        figure('relRegr1_'+str(i)+'_aggregate', figsize=(18, 11.25))
        subplot(numRowFig,numColFig,index1+1)
        
        print 'Creating figure:', 'relRegr1_'+str(i)+'_aggregate'
        '''
        # for each generation
        for j in gen:
            if j < gen[-1]:
                continue
            
            # for each repetition
            for k in range(min(len(meanRelatednessRegression1),numRep)):
                # collect the generation data of each repetition
                dataRepRelRegr1[k] = meanRelatednessRegression1[k][j][i]
                
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotDataRelRegr1[index1][i][index2][k] = dataRepRelRegr1[k]
                elif j <= gen[-1]:
                    boxPlotDataRelRegr1[index1][i][index2][k] += dataRepRelRegr1[k]
                if j == gen[-1]:
                    boxPlotDataRelRegr1[index1][i][index2][k] /= boxplotGenerations
            '''
            # for each generation, average among the repetitions
            if count_nonzero(~isnan(dataRepRelRegr1)):
                meanRelRegr1[i][j] = nanmean(dataRepRelRegr1)
                if computeStd:
                    stdRelRegr1[i][j] = nanstd(dataRepRelRegr1)
            else:
                meanRelRegr1[i][j] = np.nan
                if computeStd:
                    stdRelRegr1[i][j] = np.nan

        if computeStd:
            line = plot(gen, meanRelRegr1[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            fill_between(gen, meanRelRegr1[i]-stdRelRegr1[i], meanRelRegr1[i]+stdRelRegr1[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            tmpMin = min(meanRelRegr1[i]-stdRelRegr1[i])
            tmpMax = max(meanRelRegr1[i]+stdRelRegr1[i])
        else:
            plot(gen, meanRelRegr1[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            tmpMin = min(meanRelRegr1[i])
            tmpMax = max(meanRelRegr1[i])

        if tmpMin < minValuesRelRegr1[i]:
            minValuesRelRegr1[i] = tmpMin
        if tmpMax > maxValuesRelRegr1[i]:
            maxValuesRelRegr1[i] = tmpMax

        xlim([0,numGen])
        
        title(legend1[index1])
        if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
            xlabel('Generations')
        else:
            gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (index1+1) == 1 or (index1 % numColFig) == 0:
            if i < numGenes:
                ylabel('Relatedness ' + '(' + str(i) + ')') # 0...4 : relatedness
            elif (i+1) % 3 == 0:  # 5, 8, 11, 14, 17
                ylabel('alpha ' + '(' + str(i/3-1) + ')')
            elif (i+1) % 3 == 1:  # 6, 9, 12, 15, 18
                ylabel('-c ' + '(' + str(i/3-1) + ')')
            else:  # 7, 10, 13, 16, 19
                ylabel('b ' + '(' + str(i/3-1) + ')')
        else:
            if i < numGenes:
                gca().yaxis.set_major_formatter(plt.NullFormatter())
        
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                        numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
                
        savefig('relRegr1_'+str(i)+'_aggregate'+figureFormat)
        '''
        
    # -------------------------------------------------------------------------------------------- #
    # plot the trends for each column along the generations
    for i in range(numRelRegrMultPhen):
        '''  
        figure('relRegrMultPhen_'+str(i)+'_aggregate', figsize=(18, 11.25))
        subplot(numRowFig,numColFig,index1+1)
        
        print 'Creating figure:', 'relRegrMultPhen_'+str(i)+'_aggregate'
        '''
        # for each generation
        for j in gen:
            if j < gen[-1]:
                continue
            
            # for each repetition
            for k in range(min(len(meanRelatednessRegressionMultiPhen),numRep)):
                # collect the generation data of each repetition
                dataRepRelRegrMultPhen[k] = meanRelatednessRegressionMultiPhen[k][j][i]
                
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotDataRelRegrMultPhen[index1][i][index2][k] = dataRepRelRegrMultPhen[k]
                elif j <= gen[-1]:
                    boxPlotDataRelRegrMultPhen[index1][i][index2][k] += dataRepRelRegrMultPhen[k]
                if j == gen[-1]:
                    boxPlotDataRelRegrMultPhen[index1][i][index2][k] /= boxplotGenerations
            '''
            # for each generation, average among the repetitions
            if count_nonzero(~isnan(dataRepRelRegrMultPhen)):
                meanRelRegrMultPhen[i][j] = nanmean(dataRepRelRegrMultPhen)
                if computeStd:
                    stdRelRegrMultPhen[i][j] = nanstd(dataRepRelRegrMultPhen)
            else:
                meanRelRegrMultPhen[i][j] = np.nan
                if computeStd:
                    stdRelRegrMultPhen[i][j] = np.nan

        if computeStd:
            line = plot(gen, meanRelRegrMultPhen[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            fill_between(gen, meanRelRegrMultPhen[i]-stdRelRegrMultPhen[i], meanRelRegrMultPhen[i]+stdRelRegrMultPhen[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            tmpMin = min(meanRelRegrMultPhen[i]-stdRelRegrMultPhen[i])
            tmpMax = max(meanRelRegrMultPhen[i]+stdRelRegrMultPhen[i])
        else:
            plot(gen, meanRelRegrMultPhen[i], label=legend2[index2], color=mycolors[index2], linewidth=2)
            tmpMin = min(meanRelRegrMultPhen[i])
            tmpMax = max(meanRelRegrMultPhen[i])

        if tmpMin < minValuesRelRegrMultPhen[i]:
            minValuesRelRegrMultPhen[i] = tmpMin
        if tmpMax > maxValuesRelRegrMultPhen[i]:
            maxValuesRelRegrMultPhen[i] = tmpMax

        xlim([0,numGen])
        
        title(legend1[index1])
        if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
            xlabel('Generations')
        else:
            gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (index1+1) == 1 or (index1 % numColFig) == 0:
            if i == 0:          # 0: alpha
                ylabel('alpha')
            elif i < numGenes+1:    # 1...5: non-social gradients
                ylabel('non-social grad. ' + '(' + str(i-1) + ')')
            else:                   # 6...10: social gradients 
                ylabel('social grad. ' + '(' + str(i-numGenes-1) + ')')
        #else:
        #    gca().yaxis.set_major_formatter(plt.NullFormatter())
        
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                        numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
                
        savefig('relRegrMultPhen_'+str(i)+'_aggregate'+figureFormat)
        '''
    # -------------------------------------------------------------------------------------------- #
    # plot dispersal probability
    print "Processing pickle file", pickleFileDispProb
    with open(folder + '/' + pickleFileDispProb, 'rb') as f:
        meanDispProb = pickle.load(f)

    if firstFolder:
        # (numExperiments x numTreatments x numRep)
        boxPlotDataDispProb = np.empty((numExperiments,numTreatments,numRep))
        boxPlotDataDispProb.fill(np.nan)
    '''
    figure('disp_prob', figsize=(18, 11.25))
    subplot(numRowFig,numColFig,index1+1)
    print 'Creating figure:', 'disp_prob' 
    '''
    for j in gen:
        if j < gen[-1]:
            continue
            
        for k in range(min(len(meanAggregate),numRep)):
            # collect the generation data of each repetition
            dataRep[k] = meanDispProb[k][j]
            
            # save the last generations data for boxplot
            if j == gen[-1]-boxplotGenerations+1:
                boxPlotDataDispProb[index1][index2][k] = dataRep[k]
            elif j <= gen[-1]:
                boxPlotDataDispProb[index1][index2][k] += dataRep[k]
            if j == gen[-1]:
                boxPlotDataDispProb[index1][index2][k] /= boxplotGenerations
        '''    
        # for each generation, average among the repetitions
        mean[0][j] = nanmean(array(dataRep))
        if computeStd:
            std[0][j] = nanstd(array(dataRep))

    if computeStd:
        line = plot(gen, mean[0], label=legend2[index2], color=mycolors[index2], linewidth=2)
        fill_between(gen, mean[0]-std[0], mean[0]+std[0], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
    else:
        plot(gen, mean[0], label=legend2[index2], color=mycolors[index2], linewidth=2)

    if numTreatments > 1:
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    
    xlim([0,numGen])
    ylim(colRangesDisp)
    
    title(legend1[index1])
    if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
        xlabel('Generations')
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (index1+1) == 1 or (index1 % numColFig) == 0:
        ylabel('Dispersal probability')
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())
    
    #savefig('disp_prob'+figureFormat)
    '''
    
    # -------------------------------------------------------------------------------------------- #
    # plot group stats (occupied chambers, inter/intra group fitness variance, group size)
    '''
    print "Processing pickle file", pickleFileNrGroups
    with open(folder + '/' + pickleFileNrGroups, 'rb') as f:
        meanGroupStats = pickle.load(f)

    if firstFolder:
        numColGroupStats = len(meanGroupStats[0][0]) # number of group stats
        
        # (numExperiments x 6 x numTreatments x numRep)
        boxPlotDataGroupStats = np.empty((numExperiments,numColGroupStats,numTreatments,numRep))
        boxPlotDataGroupStats.fill(np.nan)

    for gs in range(numColGroupStats):
        figure(groupStatsFigLabels[gs], figsize=(18, 11.25))
        subplot(numRowFig,numColFig,index1+1)
        print 'Creating figure:', groupStatsFigLabels[gs]
        
        for j in gen:
            if j < gen[-1]:
                continue
            
            for k in range(min(len(meanAggregate),numRep)):
                # collect the generation data of each repetition
                dataRep[k] = meanGroupStats[k][j][gs]
                
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotDataGroupStats[index1][gs][index2][k] = dataRep[k]
                elif j <= gen[-1]:
                    boxPlotDataGroupStats[index1][gs][index2][k] += dataRep[k]
                if j == gen[-1]:
                    boxPlotDataGroupStats[index1][gs][index2][k] /= boxplotGenerations
        
            # for each generation, average among the repetitions
            mean[0][j] = nanmean(array(dataRep))
            if computeStd:
                std[0][j] = nanstd(array(dataRep))

        if computeStd:
            line = plot(gen, mean[0], label=legend2[index2], color=mycolors[index2], linewidth=2)
            fill_between(gen, mean[0]-std[0], mean[0]+std[0], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            tmpMin = min(mean[0]-std[0])
            tmpMax = max(mean[0]+std[0])
        else:
            plot(gen, mean[0], label=legend2[index2], color=mycolors[index2], linewidth=2)
            tmpMin = min(mean[0])
            tmpMax = max(mean[0])
        
        if tmpMin < minValuesStats[gs]:
            minValuesStats[gs] = tmpMin
        if tmpMax > maxValuesStats[gs]:
            maxValuesStats[gs] = tmpMax
        
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                         numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
            
        xlim([0,numGen])
        
        title(legend1[index1])
        if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
            xlabel('Generations')
        else:
            gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (index1+1) == 1 or (index1 % numColFig) == 0:
            ylabel(groupStatsYLabels[gs])
        #else:
        #    gca().yaxis.set_major_formatter(plt.NullFormatter())
        
        savefig(groupStatsFigLabels[gs]+figureFormat)
    '''
    
    # -------------------------------------------------------------------------------------------- #
    # plot group stats (occupied chambers, inter/intra group fitness variance, group size)
    # PER REPETITION
    '''
    for gs in range(numColGroupStats):
        figure(groupStatsFigLabels[gs]+'_'+legend1[index1]+'_'+legend2[index2], figsize=(18, 11.25))
        print 'Creating figure:', groupStatsFigLabels[gs]+'_'+legend1[index1]+'_'+legend2[index2]
        
        for k in range(min(len(meanAggregate),numRep)):
            line = plot(gen, meanGroupStats[k,0:numGen,gs], linewidth=2) #color=mycolors[index2], 
            xlim([0,numGen])
            title(legend1[index1]+', '+legend2[index2])
            xlabel('Generations')
            ylabel(groupStatsYLabels[gs])
            
        savefig(groupStatsFigLabels[gs]+'_'+legend1[index1]+'_'+legend2[index2]+figureFormat)
    '''
    
    # -------------------------------------------------------------------------------------------- #
    
    for rel in range(numRel):
        figure('rel-alt' + str(rel), figsize=(18, 11.25))
        subplot(numRowFig,numColFig,index1+1)
        print 'Creating figure:', 'rel-alt'+ str(rel)
        
        Y = 1
        options=folder.split('-')[1:]
        for option in options:
            if option.startswith('Y'):
                Y = float(option[1:])
        
        xdata = boxPlotDataRelatedness[index1][rel][index2]
        ydata = np.array((nanmean(boxPlotDataAggregate[index1][altruismLevelIndex][index2]),))
        
        ymin=min(ydata)
        ymax=max(ydata)
        if ymax-ymin < minVerticalHeight:
            ymin = (ymin+ymax)/2 - minVerticalHeight
            ymax = (ymin+ymax)/2 + minVerticalHeight
        
        if Y > 0:
            threshold=1.0/Y
            plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='--')
        line = plot(xdata, ydata, '.', label=legend2[index2], color=mycolors[index2], markersize=10)
        
        #xlim(colRangesRel)
        ylim(colRanges[altruismLevelIndex])
        
        title(legend1[index1])
        if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
            xlabel('Relatedness ' + '(' + str(rel) + ')')
        #else:
        #    gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (index1+1) == 1 or (index1 % numColFig) == 0:
            ylabel(columns[altruismLevelIndex])
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
    
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    
        savefig('rel-alt'+str(rel)+figureFormat)
        
    # -------------------------------------------------------------------------------------------- #
    
    for rel in range(numRelRegr1):
        if rel < numGenes:
            figure('relRegr1-alt' + str(rel), figsize=(18, 11.25))
            subplot(numRowFig,numColFig,index1+1)
            print 'Creating figure:', 'relRegr1-alt'+ str(rel)
            
            Y = 1
            options=folder.split('-')[1:]
            for option in options:
                if option.startswith('Y'):
                    Y = float(option[1:])
            
            xdata = boxPlotDataRelRegr1[index1][rel][index2]
            ydata = np.array((nanmean(boxPlotDataAggregate[index1][altruismLevelIndex][index2]),))
            
            ymin=min(ydata)
            ymax=max(ydata)
            if ymax-ymin < minVerticalHeight:
                ymin = (ymin+ymax)/2 - minVerticalHeight
                ymax = (ymin+ymax)/2 + minVerticalHeight
            
            if Y > 0:
                threshold=1.0/Y
                plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='--')
            line = plot(xdata, ydata, '.', label=legend2[index2], color=mycolors[index2], markersize=10)
            
            #0 -> 6/7
            #1 -> 9/10
            #2 -> 12/13
            #3 -> 15/16
            #4 -> 18/19
            #print rel, numGenes+rel*3+1, numGenes+rel*3+2
            
            # plot the threshold calculated as ratio of regressors (c/b)
            #threshold=np.mean(-boxPlotDataRelRegr1[index1][numGenes+rel*3+1][index2]/boxPlotDataRelRegr1[index1][numGenes+rel*3+2][index2])
            #plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='-')
            for rep in range(numRep):
                threshold=-boxPlotDataRelRegr1[index1][numGenes+rel*3+1][index2]/boxPlotDataRelRegr1[index1][numGenes+rel*3+2][index2]
                plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='-')
                
                if threshold < minValuesRelRegr1[rel]:
                    minValuesRelRegr1[rel] = threshold
                if threshold > maxValuesRelRegr1[rel]:
                    maxValuesRelRegr1[rel] = threshold
            
            #xlim(colRangesRel)
            ylim(colRanges[altruismLevelIndex])
            
            title(legend1[index1])
            if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
                xlabel('Relatedness ' + '(' + str(rel) + ')')
            #else:
            #    gca().xaxis.set_major_formatter(plt.NullFormatter())
            if (index1+1) == 1 or (index1 % numColFig) == 0:
                ylabel(columns[altruismLevelIndex])
            else:
                gca().yaxis.set_major_formatter(plt.NullFormatter())
        
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                         numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
        
            savefig('relRegr1-alt'+str(rel)+figureFormat)
    
    # -------------------------------------------------------------------------------------------- #
    
    for rel in range(numRel):
        figure('rel-disp' + str(rel), figsize=(18, 11.25))
        subplot(numRowFig,numColFig,index1+1)
        print 'Creating figure:', 'rel-disp'+ str(rel)
        
        Y = 1
        options=folder.split('-')[1:]
        for option in options:
            if option.startswith('Y'):
                Y = float(option[1:])
        
        xdata = boxPlotDataRelatedness[index1][rel][index2]
        ydata = boxPlotDataDispProb[index1][index2]
        
        ymin=min(ydata)
        ymax=max(ydata)
        if ymax-ymin < minVerticalHeight:
            ymin = (ymin+ymax)/2 - minVerticalHeight
            ymax = (ymin+ymax)/2 + minVerticalHeight
        
        if Y > 0:
            threshold=1.0/Y
            plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='--')
        line = plot(xdata, ydata, '.', label=legend2[index2], color=mycolors[index2], markersize=10)
        
        #xlim(colRangesRel)
        ylim(colRangesDisp)
        
        title(legend1[index1])
        if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
            xlabel('Relatedness ' + '(' + str(rel) + ')')
        #else:
        #    gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (index1+1) == 1 or (index1 % numColFig) == 0:
            ylabel('Dispersal probability')
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
    
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    
        savefig('rel-disp'+str(rel)+figureFormat)
    
    # -------------------------------------------------------------------------------------------- #
    
    for rel in range(numRelRegr1):
        if rel < numGenes:
            figure('relRegr1-disp' + str(rel), figsize=(18, 11.25))
            subplot(numRowFig,numColFig,index1+1)
            print 'Creating figure:', 'relRegr1-disp'+ str(rel)
            
            Y = 1
            options=folder.split('-')[1:]
            for option in options:
                if option.startswith('Y'):
                    Y = float(option[1:])
            
            xdata = boxPlotDataRelRegr1[index1][rel][index2]
            ydata = boxPlotDataDispProb[index1][index2]
            
            ymin=min(ydata)
            ymax=max(ydata)
            if ymax-ymin < minVerticalHeight:
                ymin = (ymin+ymax)/2 - minVerticalHeight
                ymax = (ymin+ymax)/2 + minVerticalHeight
            
            if Y > 0:
                threshold=1.0/Y
                plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='--')
            line = plot(xdata, ydata, '.', label=legend2[index2], color=mycolors[index2], markersize=10)
            
            #0 -> 6/7
            #1 -> 9/10
            #2 -> 12/13
            #3 -> 15/16
            #4 -> 18/19
            #print rel, numGenes+rel*3+1, numGenes+rel*3+2
            
            # plot the threshold calculated as ratio of regressors (c/b)
            #threshold=np.mean(-boxPlotDataRelRegr1[index1][numGenes+rel*3+1][index2]/boxPlotDataRelRegr1[index1][numGenes+rel*3+2][index2])
            #plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='-')
            for rep in range(numRep):
                threshold=-boxPlotDataRelRegr1[index1][numGenes+rel*3+1][index2]/boxPlotDataRelRegr1[index1][numGenes+rel*3+2][index2]
                plt.axvline(x=threshold, ymin=ymin, ymax=ymax, linewidth=2, color=mycolors[index2], linestyle='-')
                
                if threshold < minValuesRelRegr1[rel]:
                    minValuesRelRegr1[rel] = threshold
                if threshold > maxValuesRelRegr1[rel]:
                    maxValuesRelRegr1[rel] = threshold
            
            #xlim(colRangesRel)
            ylim(colRangesDisp)
            
            title(legend1[index1])
            if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
                xlabel('Relatedness ' + '(' + str(rel) + ')')
            #else:
            #    gca().xaxis.set_major_formatter(plt.NullFormatter())
            if (index1+1) == 1 or (index1 % numColFig) == 0:
                ylabel('Dispersal probability')
            else:
                gca().yaxis.set_major_formatter(plt.NullFormatter())
        
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                         numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
        
            savefig('relRegr1-disp'+str(rel)+figureFormat)
    
    # -------------------------------------------------------------------------------------------- #
    
    figure('disp-alt', figsize=(18, 11.25))
    subplot(numRowFig,numColFig,index1+1)
    print 'Creating figure:', 'disp-alt'
    
    xdata = boxPlotDataDispProb[index1][index2]
    ydata = np.array((nanmean(boxPlotDataAggregate[index1][altruismLevelIndex][index2]),))
    
    line = plot(xdata, ydata, '.', label=legend2[index2], color=mycolors[index2], markersize=10)
    
    xlim(colRangesDisp)
    ylim(colRanges[altruismLevelIndex])
    
    title(legend1[index1])
    if numColFig*numRowFig-numColFig < (index1+1) and (index1+1) <= numColFig*numRowFig:
        xlabel('Dispersal probability')
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (index1+1) == 1 or (index1 % numColFig) == 0:
        ylabel(columns[altruismLevelIndex])
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())

    handles, labels = gca().get_legend_handles_labels()
    gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                 numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)

    savefig('disp-alt'+figureFormat)
    
    firstFolder = False

# calculate the min delta on the boxplotPositions
minDelta = boxplotPositions.max()
for i in range(1,len(boxplotPositions)):
    delta = boxplotPositions[i]-boxplotPositions[i-1]
    if delta < minDelta:
        minDelta = delta

# -------------------------------------------------------------------------------------------- #
# plot the boxplot of the dispersal probability (at the last generation)

figure('disp_prob'+'_boxplot', figsize=(18, 11.25))
print 'Creating figure:', 'disp_prob'+'_boxplot'

for index in range(numExperiments):
    subplot(numRowFig,numColFig,index+1)
    
    # remove NaN
    tmpData = [x[~isnan(x)] for x in boxPlotDataDispProb[index]]
    if compare2 == "Y":
        box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
        xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
        xticks(boxplotPositions, legend2, rotation=90)
    else:
        box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
        xticks(range(1,numTreatments+1), legend2, rotation=90)
    for patch, color in zip(box['boxes'], mycolors):
        patch.set_facecolor(color)
    
    ylim(colRangesDisp)
    
    title(legend1[index])
    if numColFig*numRowFig-numColFig < (index+1) and (index+1) <= numColFig*numRowFig:
        pass
    else:
        xticks([])
    if (index+1) == 1 or (index % numColFig) == 0:
        ylabel('Dispersal probability')
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())

savefig('disp_prob'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #
# plot the boxplot of the group stats (at the last generation)
'''
for gs in range(numColGroupStats):
    figure(groupStatsFigLabels[gs]+'_boxplot', figsize=(18, 11.25))
    print 'Creating figure:', groupStatsFigLabels[gs]+'_boxplot'
    
    for index in range(numExperiments):
        subplot(numRowFig,numColFig,index+1)
        
        # remove NaN
        tmpData = [x[~isnan(x)] for x in boxPlotDataGroupStats[index][gs]]
        if compare2 == "Y":
            box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
            xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
            xticks(boxplotPositions, legend2, rotation=90)
        else:
            box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
            xticks(range(1,numTreatments+1), legend2, rotation=90)
        for patch, color in zip(box['boxes'], mycolors):
            patch.set_facecolor(color)
        
        title(legend1[index])
        if numColFig*numRowFig-numColFig < (index+1) and (index+1) <= numColFig*numRowFig:
            pass
        else:
            xticks([])
        if (index+1) == 1 or (index % numColFig) == 0:
            ylabel(groupStatsYLabels[gs])
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
        
    savefig(groupStatsFigLabels[gs]+'_boxplot' +figureFormat)
'''
# -------------------------------------------------------------------------------------------- #

# plot the boxplot for each column (at the last generation)
for i in range(numCol):
    # take only some columns
    if i not in dataIndexes:
        continue

    figure(str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    for index in range(numExperiments):
        subplot(numRowFig,numColFig,index+1)
        print 'Creating figure:', str(i)+'_aggregate'+'_boxplot'
        
        # remove NaN
        tmpData = [x[~isnan(x)] for x in boxPlotDataAggregate[index][i]]
        if compare2 == "Y":
            box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
            xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
            xticks(boxplotPositions, legend2, rotation=90)
        else:
            box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
            xticks(range(1,numTreatments+1), legend2, rotation=90)
        for patch, color in zip(box['boxes'], mycolors):
            patch.set_facecolor(color)
        
        title(legend1[index])
        if numColFig*numRowFig-numColFig < (index+1) and (index+1) <= numColFig*numRowFig:
            pass
        else:
            xticks([])
        if (index+1) == 1 or (index % numColFig) == 0:
            ylabel(columns[i])
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
        
    savefig(str(i)+'_aggregate'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #

# plot the boxplot for each column (at the last generation)
for i in range(numRel):

    figure('rel'+str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    for index in range(numExperiments):
        subplot(numRowFig,numColFig,index+1)
        print 'Creating figure:', 'rel'+str(i)+'_aggregate'+'_boxplot'
        
        # remove NaN
        tmpData = [x[~isnan(x)] for x in boxPlotDataRelatedness[index][i]]
        if compare2 == "Y":
            box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
            xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
            xticks(boxplotPositions, legend2, rotation=90)
        else:
            box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
            xticks(range(1,numTreatments+1), legend2, rotation=90)
        for patch, color in zip(box['boxes'], mycolors):
            patch.set_facecolor(color)
        
        title(legend1[index])
        if numColFig*numRowFig-numColFig < (index+1) and (index+1) <= numColFig*numRowFig:
            pass
        else:
            xticks([])
        if (index+1) == 1 or (index % numColFig) == 0:
            ylabel('Relatedness ' + '(' + str(i) + ')')
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
        
    savefig('rel'+str(i)+'_aggregate'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #

# plot the boxplot for each column (at the last generation)
for i in range(numRelRegr1):

    figure('relRegr1_'+str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    for index in range(numExperiments):
        subplot(numRowFig,numColFig,index+1)
        print 'Creating figure:', 'relRegr1_'+str(i)+'_aggregate'+'_boxplot'
        
        # remove NaN
        tmpData = [x[~isnan(x)] for x in boxPlotDataRelRegr1[index][i]]
        if compare2 == "Y":
            box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
            xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
            xticks(boxplotPositions, legend2, rotation=90)
        else:
            box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
            xticks(range(1,numTreatments+1), legend2, rotation=90)
        for patch, color in zip(box['boxes'], mycolors):
            patch.set_facecolor(color)
        
        title(legend1[index])
        if numColFig*numRowFig-numColFig < (index+1) and (index+1) <= numColFig*numRowFig:
            pass
        else:
            xticks([])
        if (index+1) == 1 or (index % numColFig) == 0:
            if i < numGenes:
                ylabel('Relatedness ' + '(' + str(i) + ')') # 0...4 : relatedness
            elif (i+1) % 3 == 0:  # 5, 8, 11, 14, 17
                ylabel('alpha ' + '(' + str(i/3-1) + ')')
            elif (i+1) % 3 == 1:  # 6, 9, 12, 15, 18
                ylabel('-c ' + '(' + str(i/3-1) + ')')
            else:  # 7, 10, 13, 16, 19
                ylabel('b ' + '(' + str(i/3-1) + ')')
        else:
            if i < numGenes:
                gca().yaxis.set_major_formatter(plt.NullFormatter())
        
    savefig('relRegr1_'+str(i)+'_aggregate'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #

# plot the boxplot for each column (at the last generation)
for i in range(numRelRegrMultPhen):

    figure('relRegrMultPhen_'+str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    for index in range(numExperiments):
        subplot(numRowFig,numColFig,index+1)
        print 'Creating figure:', 'relRegrMultPhen_'+str(i)+'_aggregate'+'_boxplot'
        
        # remove NaN
        tmpData = [x[~isnan(x)] for x in boxPlotDataRelRegrMultPhen[index][i]]
        if compare2 == "Y":
            box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
            xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
            xticks(boxplotPositions, legend2, rotation=90)
        else:
            box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
            xticks(range(1,numTreatments+1), legend2, rotation=90)
        for patch, color in zip(box['boxes'], mycolors):
            patch.set_facecolor(color)
        
        title(legend1[index])
        if numColFig*numRowFig-numColFig < (index+1) and (index+1) <= numColFig*numRowFig:
            pass
        else:
            xticks([])
        if (index+1) == 1 or (index % numColFig) == 0:
            if i == 0:          # 0: alpha
                ylabel('alpha')
            elif i < numGenes+1:    # 1...5: non-social gradients
                ylabel('non-social grad. ' + '(' + str(i-1) + ')')
            else:                   # 6...10: social gradients 
                ylabel('social grad. ' + '(' + str(i-numGenes-1) + ')')
        #else:
        #    gca().yaxis.set_major_formatter(plt.NullFormatter())
        
    savefig('relRegrMultPhen_'+str(i)+'_aggregate'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #

# adjust limits of plots
for i in range(numCol):
    # take only some columns
    if i not in dataIndexes:
        continue

    '''
    figure(str(i)+'_aggregate', figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        if colRanges[dataIndexes[i]][0] != colRanges[dataIndexes[i]][1]:
            ylim(colRanges[dataIndexes[i]])
        else:
            if minValues[dataIndexes[i]] != maxValues[dataIndexes[i]]:
                ylim([minValues[dataIndexes[i]],maxValues[dataIndexes[i]]])
    savefig(str(i)+'_aggregate'+figureFormat)
    '''
    
    figure(str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        if colRanges[dataIndexes[i]][0] != colRanges[dataIndexes[i]][1]:
            ylim(colRanges[dataIndexes[i]])
        else:
            if minValues[dataIndexes[i]] != maxValues[dataIndexes[i]]:
                ylim([minValues[dataIndexes[i]],maxValues[dataIndexes[i]]])
    savefig(str(i)+'_aggregate'+'_boxplot' +figureFormat)

'''
minRel=min(minValuesRel)
maxRel=max(maxValuesRel)

for i in range(numRel):
    figure('rel'+str(i)+'_aggregate', figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        #ylim(colRangesRel)
        if minRel != maxRel:
            ylim([minRel,maxRel])
    savefig('rel'+str(i)+'_aggregate'+figureFormat)
    
    figure('rel'+str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        #ylim(colRangesRel)
        if minRel != maxRel:
            ylim([minRel,maxRel])
    savefig('rel'+str(i)+'_aggregate'+'_boxplot' +figureFormat)

minRelRegr1=min(minValuesRelRegr1)
maxRelRegr1=max(maxValuesRelRegr1)

for i in range(numRelRegr1):
    if i < numGenes:
        figure('relRegr1_'+str(i)+'_aggregate', figsize=(18, 11.25))
        for j in range(numExperiments):
            subplot(numRowFig,numColFig,j+1)
            #ylim(colRangesRel)
            if minRelRegr1 != maxRelRegr1:
                ylim([minRelRegr1,maxRelRegr1])
        savefig('relRegr1_'+str(i)+'_aggregate'+figureFormat)
        
        figure('relRegr1_'+str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
        for j in range(numExperiments):
            subplot(numRowFig,numColFig,j+1)
            #ylim(colRangesRel)
            if minRelRegr1 != maxRelRegr1:
                ylim([minRelRegr1,maxRelRegr1])
        savefig('relRegr1_'+str(i)+'_aggregate'+'_boxplot' +figureFormat)
        
        figure('relRegr1-alt' + str(i), figsize=(18, 11.25))
        for j in range(numExperiments):
            subplot(numRowFig,numColFig,j+1)
            #xlim(colRangesRel)
            if minRelRegr1 != maxRelRegr1:
                xlim(minRelRegr1-minVerticalHeight,maxRelRegr1+minVerticalHeight)
        savefig('relRegr1-alt' + str(i)+figureFormat)
        
        figure('relRegr1-disp' + str(i), figsize=(18, 11.25))
        for j in range(numExperiments):
            subplot(numRowFig,numColFig,j+1)
            #xlim(colRangesRel)
            if minRelRegr1 != maxRelRegr1:
                xlim(minRelRegr1-minVerticalHeight,maxRelRegr1+minVerticalHeight)
        savefig('relRegr1-disp' + str(i)+figureFormat)
'''

'''
for i in range(numRelRegrMultPhen):
    figure('relRegrMultPhen_'+str(i)+'_aggregate', figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        #ylim(colRangesRel)
        if minValuesRelRegrMultPhen[i] != maxValuesRelRegrMultPhen[i]:
            ylim([minValuesRelRegrMultPhen[i],maxValuesRelRegrMultPhen[i]])
    savefig('relRegrMultPhen_'+str(i)+'_aggregate'+figureFormat)
    
    figure('relRegrMultPhen_'+str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        #ylim(colRangesRel)
        if minValuesRelRegrMultPhen[i] != maxValuesRelRegrMultPhen[i]:
            ylim([minValuesRelRegrMultPhen[i],maxValuesRelRegrMultPhen[i]])
    savefig('relRegrMultPhen_'+str(i)+'_aggregate'+'_boxplot' +figureFormat)
'''

'''
for gs in range(numColGroupStats):
    figure(groupStatsFigLabels[gs], figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        if minValuesStats[gs] != maxValuesStats[gs]:
            ylim([minValuesStats[gs],maxValuesStats[gs]])
    savefig(groupStatsFigLabels[gs]+figureFormat)
    
    figure(groupStatsFigLabels[gs]+'_boxplot', figsize=(18, 11.25))
    for j in range(numExperiments):
        subplot(numRowFig,numColFig,j+1)
        if minValuesStats[gs] != maxValuesStats[gs]:
            ylim([minValuesStats[gs],maxValuesStats[gs]])
    savefig(groupStatsFigLabels[gs]+'_boxplot' +figureFormat)
'''

# -------------------------------------------------------------------------------------------- #
#show()
