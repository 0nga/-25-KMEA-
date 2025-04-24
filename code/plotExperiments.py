# -*- coding: utf-8 -*-

import sys
import glob
import math
import os
#import cPickle as pickle
import pickle

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
# possible values: {"D", "Y", "L", "O", "SelectionTypes", "DensityInputs", "Inputs", "ZeroInputs", "AltrNonAltr", "n", "t"}
compare = sys.argv[1]

# result folder
folders = sys.argv[2:]

# max generation
#maxGen = 200
maxGen = np.inf

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

# category colors for plots of number of individuals per category
colors_categ = ['#FFA500', '#A52A2A', '#808000', '#008000', '#ADD8E6', '#0000A0'] # see http://www.computerhope.com/htmcolor.htm

# number of treatments
numTreatments=len(folders)

# legends (one per treatment)
legends = ['']*numTreatments

# consider the last generations for calculating the boxplots
boxplotPositions = np.zeros(numTreatments)
boxplotGenerations=min(maxGen,100)

# number of categories
numCategories=len(categories)

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

# data indexes for scatter plots, heatmaps, and correlation plots
dataIndex1 = fitnessIndex
dataIndex2 = altruismLevelIndex

# incremental index
n = 0

colStatsFigLabels = ['nr_groups','group_size','fitness_mean','altruism_mean','disp_prob']
colStatsYLabels = ['Nr. groups','Group size','Avg. fitness','Avg. altruism','Disp. probability']

numColStats = len(colStatsFigLabels)

colCorrRanges = [[0,0],
            [0,0],
            [0,0],
            [0,1],
            [0,1]]

groupStatsFigLabels = ['nr_groups','inter_group_diversity','intra_group_diversity','group_size_avg','group_size_min','group_size_max']
groupStatsYLabels = ['Number of groups','Inter-group fitness diversity','Intra-group fitness diversity','Group size avg', 'Group size min', 'Group size max']

# -------------------------------------------------------------------------------------------- #

# for each experiment configuration
for folder in folders:
    print("Processing folder", folder)
    
    # -------------------------------------------------------------------------------------------- #
    # create legend
    if numTreatments >= 1:
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
                    boxplotPositions[n] = float(option[1:])
        elif compare == "q":
            for option in options:
                if option.startswith('q'):
                    legends[n] = 'q='+option[1:]
        elif compare == "O":
            for option in options:
                if option.startswith('O'):
                    legends[n] = 'O='+option[1:]
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
    # load results from files
    meanCategories = {}
    for category in categories:
        print( "Processing pickle file", pickleFileCategories[category])
        with open(folder + '/' + pickleFileCategories[category], 'rb') as f:
            meanCategories[category] = pickle.load(f)

    print ("Processing pickle file", pickleFileAggregate)
    with open(folder + '/' + pickleFileAggregate, 'rb') as f:
        meanAggregate = pickle.load(f)
        
    print ("Processing pickle file", pickleFileRelatedness)
    with open(folder + '/' + pickleFileRelatedness, 'rb') as f:
        meanRelatedness = pickle.load(f)

    numRel = len(meanRelatedness[0][0])

    print ("Processing pickle file", pickleFileNrRobots)
    with open(folder + '/' + pickleFileNrRobots, 'rb') as f:
        meanNrRobotsInCategory = pickle.load(f)

    if n == 0:
        numRep = len(meanAggregate)
        numGen = min(len(meanAggregate[0]),maxGen)
        numCol = len(meanAggregate[0][0]) # number of robots' traits of interest

        # generation array
        gen = arange(0,numGen)
        
        # (numCategories x numCol x numTreatments x numRep)
        boxPlotDataCategories = np.empty((numCategories,numCol,numTreatments,numRep))
        boxPlotDataCategories.fill(np.nan)
        # (1 x numCol x numTreatments x numRep)
        boxPlotDataAggregate = np.empty((1,numCol,numTreatments,numRep))
        boxPlotDataAggregate.fill(np.nan)
        # (1 x numCategories x numTreatments x numRep) CURRENTLY NOT USED (BECAUSE OF MULTIPLE BOXPLOTS FOR EACH TREATMENT)
        boxPlotNrRobotsCategories = np.empty((1,numCategories,numTreatments,numRep))
        boxPlotNrRobotsCategories.fill(np.nan)
        
        # mean and std. dev. (over numRep)
        mean = np.empty((numCol,numGen))
        if computeStd:
            std = np.empty((numCol,numGen))
        # single repetition data
        dataRep = np.empty(numRep)
        dataRep.fill(np.nan)

        # (numCol x numRep x numGen)
        dataAggregate = np.empty((numCol,numRep,numGen))
        dataAggregate.fill(np.nan)

        minValues = np.empty(numCol)
        minValues.fill(np.inf)
        maxValues = np.empty(numCol)
        maxValues.fill(-np.inf)
        
        # (1 x numRel x numTreatments x numRep)
        boxPlotDataRelatedness = np.empty((1,numRel,numTreatments,numRep))
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

    # plot the nr of individuals per category (averaged over all the repetitions)
    figure('robots', figsize=(18, 11.25))
    print( 'Creating figure:', 'robots')
    
    subplot(numRowFig,numColFig,n+1)
    
    # for each category
    for i in range(numCategories):
        # for each generation
        for j in gen:
            # for each repetition
            for k in range(min(len(meanNrRobotsInCategory),numRep)):
                #print np.sum(meanNrRobotsInCategory[k][j][:]) # TODO check that total nr. of robots is consistent
                # collect the generation data of each repetition
                dataRep[k] = (meanNrRobotsInCategory[k][j][i]/np.sum(meanNrRobotsInCategory[k][j][:]))*100
            
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotNrRobotsCategories[0][i][n][k] = dataRep[k]
                elif j <= gen[-1]:
                    boxPlotNrRobotsCategories[0][i][n][k] += dataRep[k]
                if j == gen[-1]:
                    boxPlotNrRobotsCategories[0][i][n][k] /= boxplotGenerations
            
            if count_nonzero(~isnan(dataRep)):
                mean[i][j] = nanmean(dataRep)
                if computeStd:
                    std[i][j] = nanstd(dataRep)

        if computeStd:
            line = plot(gen, mean[i], color=colors_categ[i], linewidth=2, label=categoryNames[i])
            fill_between(gen, mean[i]-std[i], mean[i]+std[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
        else:
            plot(gen, mean[i], color=colors_categ[i], linewidth=2, label=categoryNames[i])
    
    if numTreatments > 1:      
        title(legends[n])
    
    handles, labels = gca().get_legend_handles_labels()
    gca().legend(handles, labels, bbox_to_anchor=(0.5, 0.01), bbox_transform=gcf().transFigure, loc='lower center',
                 numpoints=1, prop={'size':legendSize}, ncol=len(categoryNames), fancybox=True, shadow=True, borderaxespad=0.)          

    xlim([0,numGen])
    ylim([0,100])
    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        xlabel('Generations')
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel('Robots (%)')
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())
    savefig('robots'+figureFormat)

    # plot the trends for each column along the generations
    for i in range(numCol):
        # take only some columns
        if i not in dataIndexes:
            continue

        # -------------------------------------------------------------------------------------------- #

        fig=figure(str(i)+'_separate', figsize=(18, 11.25))
        print ('Creating figure:', str(i)+'_separate')
        
        fig.text(0.04, 0.5, columns[i], va='center', rotation='vertical')
        
        # for each category
        for l in range(numCategories):
            ax = subplot(numCategories,1,l+1)
            # for each generation
            for j in gen:
                # for each repetition
                for k in range(min(len(meanNrRobotsInCategory),numRep)):
                    # collect the generation data of each repetition
                    dataRep[k] = meanCategories[categories[l]][k][j][i]
                    
                    # save the last generations data for boxplot
                    if j == gen[-1]-boxplotGenerations+1:
                        boxPlotDataCategories[l][i][n][k] = dataRep[k]
                    elif j <= gen[-1]:
                        boxPlotDataCategories[l][i][n][k] += dataRep[k]
                    if j == gen[-1]:
                        boxPlotDataCategories[l][i][n][k] /= boxplotGenerations
            
                # for each generation, average among the repetitions
                if count_nonzero(~isnan(dataRep)) > 0:
                    mean[i][j] = nanmean(dataRep)
                    if computeStd:
                        std[i][j] = nanstd(dataRep)
                else:
                    mean[i][j] = np.nan
                    if computeStd:
                        std[i][j] = np.nan
            
            if computeStd:
                line = plot(gen, mean[i], label=legends[n], color=mycolors[n], linewidth=2)
                fill_between(gen, mean[i]-std[i], mean[i]+std[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
                tmpMin = min(mean[i]-std[i])
                tmpMax = max(mean[i]+std[i])
            else:
                line = plot(gen, mean[i], label=legends[n], color=mycolors[n], linewidth=2)
                tmpMin = min(mean[i])
                tmpMax = max(mean[i])
        
            if tmpMin < minValues[i]:
                minValues[i] = tmpMin
            if tmpMax > maxValues[i]:
                maxValues[i] = tmpMax

            if l == numCategories-1:
                xlabel('Generations')
            else:
                gca().xaxis.set_major_formatter(plt.NullFormatter())
            title(categoryNames[l])
            if numTreatments > 1:
                handles, labels = gca().get_legend_handles_labels()
                gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                             numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
            
        # update the y range of all the subplots
        for l1 in range(numCategories):
            ax = subplot(numCategories,1,l1+1)
            if colRanges[i][0] != colRanges[i][1]:
                ylim(colRanges[i])
            else:
                if minValues[i] != maxValues[i]:
                    ylim([minValues[i],maxValues[i]])
        savefig(str(i)+'_separate'+figureFormat)

        # -------------------------------------------------------------------------------------------- #

        fig=figure(str(i)+'_separate2', figsize=(18, 11.25))
        print ('Creating figure:', str(i)+'_separate2')
        
        subplot(numRowFig,numColFig,n+1)
        
        # for each category
        for l in range(numCategories):
            # for each generation
            for j in gen:
                # for each repetition
                for k in range(min(len(meanNrRobotsInCategory),numRep)):
                    # collect the generation data of each repetition
                    dataRep[k] = meanCategories[categories[l]][k][j][i]
                    
                    # save the last generations data for boxplot
                    if j == gen[-1]-boxplotGenerations+1:
                        boxPlotDataCategories[l][i][n][k] = dataRep[k]
                    elif j <= gen[-1]:
                        boxPlotDataCategories[l][i][n][k] += dataRep[k]
                    if j == gen[-1]:
                        boxPlotDataCategories[l][i][n][k] /= boxplotGenerations
            
                # for each generation, average among the repetitions
                if count_nonzero(~isnan(dataRep)) > 0:
                    mean[i][j] = nanmean(dataRep)
                    if computeStd:
                        std[i][j] = nanstd(dataRep)
                else:
                    mean[i][j] = np.nan
                    if computeStd:
                        std[i][j] = np.nan
            
            if computeStd:
                line = plot(gen, mean[i], color=colors_categ[l], linewidth=2, label=categoryNames[l])
                fill_between(gen, mean[i]-std[i], mean[i]+std[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
                tmpMin = min(mean[i]-std[i])
                tmpMax = max(mean[i]+std[i])
            else:
                line = plot(gen, mean[i], color=colors_categ[l], linewidth=2, label=categoryNames[l])
                tmpMin = min(mean[i])
                tmpMax = max(mean[i])
        
            if tmpMin < minValues[i]:
                minValues[i] = tmpMin
            if tmpMax > maxValues[i]:
                maxValues[i] = tmpMax
                
        if numTreatments > 1:
            title(legends[n])

        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(0.5, 0.01), bbox_transform=gcf().transFigure, loc='lower center',
                         numpoints=1, prop={'size':legendSize}, ncol=len(categoryNames), fancybox=True, shadow=True, borderaxespad=0.)            
        
        xlim([0,numGen])

        if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
            xlabel('Generations')
        else:
            gca().xaxis.set_major_formatter(plt.NullFormatter())
        if (n+1) == 1 or (n % numColFig) == 0:
            ylabel(columns[i])
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
            
        # update the y range of all the subplots
        for n1 in range(numTreatments):
            ax = subplot(numRowFig,numColFig,n1+1)
            if colRanges[i][0] != colRanges[i][1]:
                ylim(colRanges[i])
            else:
                if minValues[i] != maxValues[i]:
                    ylim([minValues[i],maxValues[i]])
        savefig(str(i)+'_separate2'+figureFormat)

        # -------------------------------------------------------------------------------------------- #

        figure(str(i)+'_aggregate', figsize=(18, 11.25))
        print( 'Creating figure:', str(i)+'_aggregate')
        
        # for each generation
        for j in gen:
            # for each repetition
            for k in range(min(len(meanNrRobotsInCategory),numRep)):
                # collect the generation data of each repetition
                dataRep[k] = meanAggregate[k][j][i]
                
                # fill data structure
                dataAggregate[i][k][j] = dataRep[k]
                
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotDataAggregate[0][i][n][k] = dataRep[k]
                elif j <= gen[-1]:
                    boxPlotDataAggregate[0][i][n][k] += dataRep[k]
                if j == gen[-1]:
                    boxPlotDataAggregate[0][i][n][k] /= boxplotGenerations
        
            # for each generation, average among the repetitions
            if count_nonzero(~isnan(dataRep)):
                mean[i][j] = nanmean(dataRep)
                if computeStd:
                    std[i][j] = nanstd(dataRep)
            else:
                mean[i][j] = np.nan
                if computeStd:
                    std[i][j] = np.nan

        if computeStd:
            line = plot(gen, mean[i], label=legends[n], color=mycolors[n], linewidth=2)
            fill_between(gen, mean[i]-std[i], mean[i]+std[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
        else:
            plot(gen, mean[i], label=legends[n], color=mycolors[n], linewidth=2)

        xlabel('Generations')
        ylabel(columns[i])
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                        numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
        if colRanges[i][0] != colRanges[i][1]:
            ylim(colRanges[i])
        savefig(str(i)+'_aggregate'+figureFormat)
        
    # -------------------------------------------------------------------------------------------- #
    # plot the trends for each column along the generations
    for i in range(numRel):
                
        figure('rel'+str(i)+'_aggregate', figsize=(18, 11.25))
        print( 'Creating figure:', 'rel'+str(i)+'_aggregate')
        
        # for each generation
        for j in gen:
            # for each repetition
            for k in range(min(len(meanRelatedness),numRep)):
                # collect the generation data of each repetition
                dataRepRel[k] = meanRelatedness[k][j][i]
                
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotDataRelatedness[0][i][n][k] = dataRepRel[k]
                elif j <= gen[-1]:
                    boxPlotDataRelatedness[0][i][n][k] += dataRepRel[k]
                if j == gen[-1]:
                    boxPlotDataRelatedness[0][i][n][k] /= boxplotGenerations
        
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
            line = plot(gen, meanRel[i], label=legends[n], color=mycolors[n], linewidth=2)
            fill_between(gen, meanRel[i]-stdRel[i], meanRel[i]+stdRel[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            tmpMin = min(meanRel[i]-stdRel[i])
            tmpMax = max(meanRel[i]+stdRel[i])
        else:
            plot(gen, meanRel[i], label=legends[n], color=mycolors[n], linewidth=2)
            tmpMin = min(meanRel[i])
            tmpMax = max(meanRel[i])

        if tmpMin < minValuesRel[i]:
            minValuesRel[i] = tmpMin
        if tmpMax > maxValuesRel[i]:
            maxValuesRel[i] = tmpMax

        xlim([0,numGen])
        
        xlabel('Generations')
        ylabel('Relatedness ' + '(' + str(i) + ')')
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                        numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
                
        savefig('rel'+str(i)+'_aggregate'+figureFormat)
    
    # -------------------------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------------------------- #
    # plot food efficiency
    '''
    print "Processing pickle file", pickleFileForaging
    with open(folder + '/' + pickleFileForaging, 'rb') as f:
        meanForaging = pickle.load(f)

    if n == 0:
        # (1 x 1 x numTreatments x numRep)
        boxPlotDataFoodForaging = np.empty((1,1,numTreatments,numRep))
        boxPlotDataFoodForaging.fill(np.nan)

    figure('food_efficiency', figsize=(18, 11.25))
    print 'Creating figure:', 'food_efficiency' 
    
    for j in gen:
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            # collect the generation data of each repetition
            dataRep[k] = meanForaging[k][j]
            
            # save the last generations data for boxplot
            if j == gen[-1]-boxplotGenerations+1:
                boxPlotDataFoodForaging[0][0][n][k] = dataRep[k]
            elif j <= gen[-1]:
                boxPlotDataFoodForaging[0][0][n][k] += dataRep[k]
            if j == gen[-1]:
                boxPlotDataFoodForaging[0][0][n][k] /= boxplotGenerations

        # for each generation, average among the repetitions
        mean[0][j] = nanmean(array(dataRep))
        if computeStd:
            std[0][j] = nanstd(array(dataRep))

    if computeStd:
        line = plot(gen, mean[0], label=legends[n], color=mycolors[n], linewidth=2)
        fill_between(gen, mean[0]-std[0], mean[0]+std[0], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
    else:
        plot(gen, mean[0], label=legends[n], color=mycolors[n])

    if numTreatments > 1:
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    xlabel('Generations')
    ylabel('Foraging efficiency')
    savefig('food_efficiency'+figureFormat)
    '''

    # -------------------------------------------------------------------------------------------- #
    # plot dispersal probability
    print( "Processing pickle file", pickleFileDispProb)
    with open(folder + '/' + pickleFileDispProb, 'rb') as f:
        meanDispProb = pickle.load(f)

    if n == 0:
        # (1 x 1 x numTreatments x numRep)
        boxPlotDataDispProb = np.empty((1,1,numTreatments,numRep))
        boxPlotDataDispProb.fill(np.nan)

    figure('disp_prob', figsize=(18, 11.25))
    print ('Creating figure:', 'disp_prob' )
    
    for j in gen:
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            # collect the generation data of each repetition
            dataRep[k] = meanDispProb[k][j]
            
            # save the last generations data for boxplot
            if j == gen[-1]-boxplotGenerations+1:
                boxPlotDataDispProb[0][0][n][k] = dataRep[k]
            elif j <= gen[-1]:
                boxPlotDataDispProb[0][0][n][k] += dataRep[k]
            if j == gen[-1]:
                boxPlotDataDispProb[0][0][n][k] /= boxplotGenerations
                
        # for each generation, average among the repetitions
        mean[0][j] = nanmean(array(dataRep))
        if computeStd:
            std[0][j] = nanstd(array(dataRep))

    if computeStd:
        line = plot(gen, mean[0], label=legends[n], color=mycolors[n], linewidth=2)
        fill_between(gen, mean[0]-std[0], mean[0]+std[0], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
    else:
        plot(gen, mean[0], label=legends[n], color=mycolors[n], linewidth=2)

    if numTreatments > 1:
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    xlabel('Generations')
    ylabel('Dispersal probability')
    ylim(colRangesDisp)
    savefig('disp_prob'+figureFormat)

    # -------------------------------------------------------------------------------------------- #
    # plot group stats (occupied chambers, inter/intra group fitness variance, group size)(
    print ("Processing pickle file", pickleFileNrGroups)
    with open(folder + '/' + pickleFileNrGroups, 'rb') as f:
        meanGroupStats = pickle.load(f)

    if n == 0:
        numColGroupStats = len(meanGroupStats[0][0]) # number of group stats
        
        # (1 x 6 x numTreatments x numRep)
        boxPlotDataGroupStats = np.empty((1,numColGroupStats,numTreatments,numRep))
        boxPlotDataGroupStats.fill(np.nan)
    
    for gs in range(numColGroupStats):
        figure(groupStatsFigLabels[gs], figsize=(18, 11.25))
        print( 'Creating figure:', groupStatsFigLabels[gs])
        
        for j in gen:
            for k in range(min(len(meanNrRobotsInCategory),numRep)):
                # collect the generation data of each repetition
                dataRep[k] = meanGroupStats[k][j][gs]
                
                # save the last generations data for boxplot
                if j == gen[-1]-boxplotGenerations+1:
                    boxPlotDataGroupStats[0][gs][n][k] = dataRep[k]
                elif j <= gen[-1]:
                    boxPlotDataGroupStats[0][gs][n][k] += dataRep[k]
                if j == gen[-1]:
                    boxPlotDataGroupStats[0][gs][n][k] /= boxplotGenerations
        
            # for each generation, average among the repetitions
            mean[0][j] = nanmean(array(dataRep))
            if computeStd:
                std[0][j] = nanstd(array(dataRep))

        if computeStd:
            line = plot(gen, mean[0], label=legends[n], color=mycolors[n], linewidth=2)
            fill_between(gen, mean[0]-std[0], mean[0]+std[0], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
        else:
            plot(gen, mean[0], label=legends[n], color=mycolors[n], linewidth=2)
        
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                         numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)

        xlabel('Generations')
        ylabel(groupStatsYLabels[gs])
        savefig(groupStatsFigLabels[gs]+figureFormat)

    # -------------------------------------------------------------------------------------------- #
    # individual repetitions plots along all generations (on aggregate values)
    figure('repetitions_'+legends[n], figsize=(18, 11.25))
    print ('Creating figure:', 'repetitions_'+legends[n])
    
    mycolors_rep = mycolors
    genRep = gen
    
    for i in range(nPlots):
        ax = subplot(nPlots+2,1,i+1)
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            plot(genRep, dataAggregate[dataIndexes[i]][k][genRep], color=mycolors_rep[n])
        plot(genRep, nanmean(array(dataAggregate[dataIndexes[i]]),axis=0)[genRep], 'k', linewidth=2)
        gca().xaxis.set_major_formatter(plt.NullFormatter())
        ylabel(columns[dataIndexes[i]], rotation='horizontal', labelpad=70)
    
    # plot dispersal probability and group size
    i=i+1
    ax = subplot(nPlots+2,1,i+1)
    for j in genRep:
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            # collect the generation data of each repetition
            dataRep[k] = meanDispProb[k][j]  
        # for each generation, average among the repetitions
        mean[0][j] = nanmean(array(dataRep))
    for k in range(min(len(meanNrRobotsInCategory),numRep)):
        plot(genRep, meanDispProb[k][genRep], color=mycolors_rep[n])
    plot(genRep, mean[0][genRep], 'k', linewidth=2)  
    gca().xaxis.set_major_formatter(plt.NullFormatter())
    ylabel('Dispersal probability', rotation='horizontal', labelpad=70)  
    
    i=i+1
    ax = subplot(nPlots+2,1,i+1)
    rep = np.empty((numRep,numGen))
    for j in genRep:
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            # collect the generation data of each repetition
            dataRep[k] = meanGroupStats[k][j][3]
            rep[k][j] = dataRep[k]
        # for each generation, average among the repetitions
        mean[0][j] = nanmean(array(dataRep))
    for k in range(min(len(meanNrRobotsInCategory),numRep)):
        plot(genRep, rep[k][genRep], color=mycolors_rep[n])
    plot(genRep, mean[0][genRep], 'k', linewidth=2)
    ylabel('Group size', rotation='horizontal', labelpad=70)  
    
    suptitle(legends[n])

    xlabel('Generations')
    savefig('repetitions_'+legends[n]+figureFormat)

    '''
    # plots (averaged over all repetitions values) along all generations (on aggregate values)
    figure('repetitions_avg_'+legends[n], figsize=(18, 11.25))
    print 'Creating figure:', 'repetitions_avg_'+legends[n]
    
    for i in range(nPlots):
        ax = subplot(nPlots,1,i+1)
        plot(gen, nanmean(array(dataAggregate[dataIndexes[i]]),axis=0), color=mycolors[n])
        ylabel(columns[dataIndexes[i]], rotation='horizontal', labelpad=70)
    suptitle(legends[n])
    xlabel('Generations')
    savefig('repetitions_avg_'+legends[n]+figureFormat)
    '''

    '''
    # scatter plot/heatmap data (on aggregate values)
    xdata = np.empty(numRep*numGen)
    ydata = np.empty(numRep*numGen)
    for k in range(min(len(meanNrRobotsInCategory),numRep)):
        for j in gen:
            xdata[k*numGen+j] = dataAggregate[dataIndex1][k][j]
            ydata[k*numGen+j] = dataAggregate[dataIndex2][k][j]
    xdata = array(xdata)
    ydata = array(ydata)

    # mask NaN
    xdata = np.ma.array(xdata, mask=np.isnan(xdata))
    xdata = np.ma.array(xdata, mask=np.isnan(ydata))
    ydata = np.ma.array(ydata, mask=np.isnan(xdata))
    ydata = np.ma.array(ydata, mask=np.isnan(ydata))

    mask = ~np.isnan(xdata) & ~np.isnan(ydata)
    xdata = xdata[mask]
    ydata = ydata[mask]
    '''

    # -------------------------------------------------------------------------------------------- #
    # heatmaps (for all the repetitions along the generations)
    '''
    figure('heatmap', figsize=(18, 11.25))
    print 'Creating figure:', 'heatmap'
    
    subplot(numRowFig,numColFig,n+1)
    hist2d(xdata, ydata, bins=[50,50], norm=colors.LogNorm())
    # colorbar(orientation='horizontal')
    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        xlabel(columns[dataIndex1])
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel(columns[dataIndex2])
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())
    if colRanges[dataIndex1][0] != colRanges[dataIndex1][1]:
        xlim(colRanges[dataIndex1])
    if colRanges[dataIndex2][0] != colRanges[dataIndex2][1]:
        ylim(colRanges[dataIndex2])
    if numTreatments > 1:
        title(legends[n])
    savefig('heatmap'+figureFormat)
    '''

    # -------------------------------------------------------------------------------------------- #
    # scatter plot and linear regression (for all the repetitions along the generations)
    '''
    figure('scatter', figsize=(18, 11.25))
    print 'Creating figure:', 'scatter'
    
    subplot(numRowFig,numColFig,n+1)
    line = plot(xdata, ydata, '.', label=legends[n], color=mycolors[n])
    if line is not None:
        mask = ~np.isnan(xdata) & ~np.isnan(ydata)
        if len(xdata[mask]) > 0 and len(ydata[mask]) > 0 and min(xdata[mask]) != max(xdata[mask]):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(xdata[mask], ydata[mask])
                # corrcoeff = np.corrcoef(xdata, ydata)[0][1] #r_value is equivalent to corrcoeff
                #print slope, intercept, r_value, p_value, std_err
                x = np.linspace(np.min(xdata), np.max(xdata), 50)
                y = [slope*x_+intercept for x_ in x]
                plot(x, y, '-', color='k',linewidth=2) #color=line[0].get_color()
            except RuntimeWarning:
                print ''
    line = None

    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        xlabel(columns[dataIndex1])
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel(columns[dataIndex2])
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())
    if colRanges[dataIndex1][0] != colRanges[dataIndex1][1]:
        xlim(colRanges[dataIndex1])
    if colRanges[dataIndex2][0] != colRanges[dataIndex2][1]:
        ylim(colRanges[dataIndex2])
    if numTreatments > 1:
        title(legends[n])
    savefig('scatter'+figureFormat)
    '''
    
    # -------------------------------------------------------------------------------------------- #    
    for rel in range(numRel):
        figure('rel-alt' + str(rel), figsize=(18, 11.25))
        print ('Creating figure:', 'rel-alt'+ str(rel))
        
        Y = 1
        options=folder.split('-')[1:]
        for option in options:
            if option.startswith('Y'):
                Y = float(option[1:])
        
        xdata = boxPlotDataRelatedness[0][rel][n]
        ydata = boxPlotDataAggregate[0][altruismLevelIndex][n]
        
        if Y > 0:
            threshold=1.0/Y
            plt.axvline(x=threshold, ymin=min(ydata), ymax = max(ydata), linewidth=1, color=mycolors[n], linestyle='--')
            #line = plot(xdata[xdata<=threshold], ydata[xdata<=threshold], '.', label=legends[n], color=mycolors[n], markersize=10)
            #line = plot(xdata[xdata>threshold], ydata[xdata>threshold], 'o', label=legends[n], color=mycolors[n], markersize=10)
            line = plot(xdata, ydata, '.', label=legends[n], color=mycolors[n], markersize=10)
        else:
            line = plot(xdata, ydata, '.', label=legends[n], color=mycolors[n], markersize=10)
        
        xlim(colRangesRel)
        ylim(colRanges[altruismLevelIndex])
        
        xlabel('Relatedness ' + '(' + str(rel) + ')')
        ylabel(columns[altruismLevelIndex])
        
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    
        savefig('rel-alt'+str(rel)+figureFormat)
    
    # -------------------------------------------------------------------------------------------- #
    # individual repetitions plots along all generations (on aggregate values) at last G generations
    
    # take only last G generations
    G=50    
    
    genRep = arange(gen[-1]+1-G,gen[-1]+1)
        
    # color map
    NUM_COLORS_rep = numRep
    cm_rep = plt.get_cmap(colorMap)
    values_rep = range(NUM_COLORS_rep)
    cNorm_rep = colors.Normalize(vmin=values_rep[0], vmax=values_rep[-1])
    scalarMap_rep = cmx.ScalarMappable(norm=cNorm_rep, cmap=cm_rep)
    
    #scalarMap_rep = [cm_rep(1.*v/NUM_COLORS_rep) for v in values_rep]    # manual scaling
    mycolors_rep = [scalarMap_rep.to_rgba(v) for v in values_rep]   # map scaling
    genRep = arange(gen[-1]+1-G,gen[-1]+1)
    
    figure('repetitions_'+str(G)+'g_'+legends[n], figsize=(18, 11.25))
    print ('Creating figure:', 'repetitions_'+str(G)+'g_'+legends[n])
    
    for i in range(nPlots):
        ax = subplot(nPlots+2,1,i+1)
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            plot(genRep, dataAggregate[dataIndexes[i]][k][genRep], color=mycolors_rep[k]) #color=mycolors_rep[n])
        plot(genRep, nanmean(array(dataAggregate[dataIndexes[i]]),axis=0)[genRep], 'k', linewidth=2)
        gca().xaxis.set_major_formatter(plt.NullFormatter())
        ylabel(columns[dataIndexes[i]], rotation='horizontal', labelpad=70)
    
    # plot dispersal probability and group size
    i=i+1
    ax = subplot(nPlots+2,1,i+1)
    for j in genRep:
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            # collect the generation data of each repetition
            dataRep[k] = meanDispProb[k][j]  
        # for each generation, average among the repetitions
        mean[0][j] = nanmean(array(dataRep))
    for k in range(min(len(meanNrRobotsInCategory),numRep)):
        plot(genRep, meanDispProb[k][genRep], color=mycolors_rep[k]) #color=mycolors_rep[n])
    plot(genRep, mean[0][genRep], 'k', linewidth=2)  
    gca().xaxis.set_major_formatter(plt.NullFormatter())
    ylabel('Dispersal probability', rotation='horizontal', labelpad=70)  
    
    i=i+1
    ax = subplot(nPlots+2,1,i+1)
    rep = np.empty((numRep,numGen))
    for j in genRep:
        for k in range(min(len(meanNrRobotsInCategory),numRep)):
            # collect the generation data of each repetition
            dataRep[k] = meanGroupStats[k][j][3]
            rep[k][j] = dataRep[k]
        # for each generation, average among the repetitions
        mean[0][j] = nanmean(array(dataRep))
    for k in range(min(len(meanNrRobotsInCategory),numRep)):
        plot(genRep, rep[k][genRep], color=mycolors_rep[k]) #color=mycolors_rep[n])
    plot(genRep, mean[0][genRep], 'k', linewidth=2)
    ylabel('Group size', rotation='horizontal', labelpad=70)  
    
    suptitle(legends[n])
    
    xlabel('Generations')
    savefig('repetitions_'+str(G)+'g_'+legends[n]+figureFormat)
    
    figure('correlation_aggregate_'+str(G)+'g_'+legends[n], figsize=(18, 11.25)) 
    print ('Creating figure:', 'correlation_aggregate_'+str(G)+'g_'+legends[n])
    f = 0
    
    nrGroups = np.empty(G)
    groupSize = np.empty(G)
    dispProb = np.empty(G)
                
    for gs1 in range(numColStats):
        for gs2 in range(numColStats):
            #if True:
            #    ax=subplot(numColStats,numColStats,f+1)
            if gs1 > gs2:
                ax=subplot(numColStats-1,numColStats-1,f+1-numColStats-gs1+1)
                
                ax.locator_params(axis='both',tight=False,nbins=4)
                
                for k in range(min(len(meanNrRobotsInCategory),numRep)):
                    for j in range(G):
                        # collect the generation data of each repetition
                        nrGroups[j] = meanGroupStats[k][genRep[j]][0]
                        groupSize[j] = meanGroupStats[k][genRep[j]][3]
                        dispProb[j] = meanDispProb[k][genRep[j]]
                    
                    if gs2 == 0:
                        xdata = nrGroups
                    elif gs2 == 1:
                        xdata = groupSize
                    elif gs2 == 2:
                        xdata = dataAggregate[fitnessIndex][k][genRep]
                    elif gs2 == 3:
                        xdata = dataAggregate[altruismLevelIndex][k][genRep]
                    elif gs2 == 4:
                        xdata = dispProb
                    
                    if gs1 == 0:
                        ydata = nrGroups
                    elif gs1 == 1:
                        ydata = groupSize
                    elif gs1 == 2:
                        ydata = dataAggregate[fitnessIndex][k][genRep]
                    elif gs1 == 3:
                        ydata = dataAggregate[altruismLevelIndex][k][genRep]
                    elif gs1 == 4:
                        ydata = dispProb
                                
                    # mask NaN
                    xdata = np.ma.array(xdata, mask=np.isnan(xdata))
                    xdata = np.ma.array(xdata, mask=np.isnan(ydata))
                    ydata = np.ma.array(ydata, mask=np.isnan(xdata))
                    ydata = np.ma.array(ydata, mask=np.isnan(ydata))
                
                    line = plot(xdata, ydata, '.', label=legends[n], color=mycolors_rep[k]) #color=mycolors_rep[n])
                    
                    '''
                    if line is not None:
                        mask = ~np.isnan(xdata) & ~np.isnan(ydata)
                        if len(xdata[mask]) > 0 and len(ydata[mask]) > 0 and min(xdata[mask]) != max(xdata[mask]):
                            try:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(xdata[mask], ydata[mask])
                                #print slope, intercept, r_value, p_value, std_err
                                if not sys.version_info[:2] == (2,6):
                                    note = 'r='+"{:g}".format(r_value)+'\n'+'p='+"{:g}".format(p_value)
                                else:
                                    note = 'r='+str(r_value)+'\n'+'p='+str(p_value)
                                #text(0.5, 0.75, note, bbox=dict(facecolor='w', edgecolor='k'), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                                plot(xdata[mask], [slope*x+intercept for x in xdata[mask]],color='k',linewidth=1)
                            except RuntimeWarning:
                                print ''
                    line = None
                    '''
                    
                    minXdataK = min(xdata)
                    maxXdataK = max(xdata)
                    minYdataK = min(ydata)
                    maxYdataK = max(ydata)
                    
                    if k == 0 or minXdataK < minXdata:
                        minXdata = minXdataK
                    if k == 0 or maxXdataK > maxXdata:
                        maxXdata = maxXdataK
                    if k == 0 or minYdataK < minYdata:
                        minYdata = minYdataK
                    if k == 0 or maxYdataK > maxYdata:
                        maxYdata = maxYdataK
                
                if (numColStats*numColStats-numColStats) < (f+1) and (f+1) <= (numColStats*numColStats):
                    xlabel(colStatsYLabels[gs2])
                else:
                    gca().xaxis.set_major_formatter(plt.NullFormatter())
                if f == 0 or (f % numColStats) == 0:
                    ylabel(colStatsYLabels[gs1])
                else:
                    gca().yaxis.set_major_formatter(plt.NullFormatter())
                
                if colCorrRanges[gs2][0] != colCorrRanges[gs2][1]:
                    xlim(colCorrRanges[gs2])
                else:
                    #print minXdata, maxXdata, maxXdata-minXdata < 1
                    if maxXdata-minXdata < 1:
                        gca().set_autoscale_on(False)
                        xlim([(minXdata+maxXdata)/2-0.5,(minXdata+maxXdata)/2+0.5])
                if colCorrRanges[gs1][0] != colCorrRanges[gs1][1]:
                    ylim(colCorrRanges[gs1])
                else:
                    #print minYdata, maxYdata, maxYdata-minYdata < 1
                    if maxYdata-minYdata < 1:
                        gca().set_autoscale_on(False)
                        ylim([(minYdata+maxYdata)/2-0.5,(minYdata+maxYdata)/2+0.5])
                savefig('correlation_aggregate_'+str(G)+'g_'+legends[n]+figureFormat)
                
            f = f+1

    n = n+1

# calculate the min delta on the boxplotPositions
#boxplotPositions=boxplotPositions.sort()
minDelta = boxplotPositions.max()
for i in range(1,len(boxplotPositions)):
    delta = boxplotPositions[i]-boxplotPositions[i-1]
    if delta < minDelta:
        minDelta = delta

# -------------------------------------------------------------------------------------------- #

# adjust limits of repetitions plots
for n in range(numTreatments):

    figure('repetitions_'+legends[n], figsize=(18, 11.25))
    print( 'Creating figure:', 'repetitions_'+legends[n])
    
    for i in range(nPlots):
        ax = subplot(nPlots+2,1,i+1)
        if colRanges[dataIndexes[i]][0] != colRanges[dataIndexes[i]][1]:
            ylim(colRanges[dataIndexes[i]])
        else:
            if minValues[dataIndexes[i]] != maxValues[dataIndexes[i]]:
                ylim([minValues[dataIndexes[i]],maxValues[dataIndexes[i]]])
 
    i=i+1
    ax = subplot(nPlots+2,1,i+1)
    ylim([0,1])
    i=i+1
    ax = subplot(nPlots+2,1,i+1)
    savefig('repetitions_'+legends[n]+figureFormat)

    '''
    figure('repetitions_avg_'+legends[n], figsize=(18, 11.25))
    print 'Creating figure:', 'repetitions_avg_'+legends[n]
    
    title(legends[n])
    for i in range(nPlots):
        ax = subplot(nPlots,1,i+1)
        if colRanges[dataIndexes[i]][0] != colRanges[dataIndexes[i]][1]:
            ylim(colRanges[dataIndexes[i]])
        else:
            if minValues[dataIndexes[i]] != maxValues[dataIndexes[i]]:
                ylim([minValues[dataIndexes[i]],maxValues[dataIndexes[i]]])
    savefig('repetitions_avg_'+legends[n]+figureFormat)
    '''

# -------------------------------------------------------------------------------------------- #
# plot the boxplot of the food efficiency (at the last generation)
'''
figure('food_efficiency'+'_boxplot', figsize=(18, 11.25))
print 'Creating figure:', 'food_efficiency'+'_boxplot'

# remove NaN
tmpData = [x[~isnan(x)] for x in boxPlotDataFoodForaging[0][0]]
if compare == "Y":
    box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
    xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
    xticks(boxplotPositions, legends)
else:
    box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
    xticks(range(1,numTreatments+1), legends)
for patch, color in zip(box['boxes'], mycolors):
    patch.set_facecolor(color)
ylabel('Foraging efficiency')
savefig('food_efficiency'+'_boxplot' +figureFormat)
'''

# -------------------------------------------------------------------------------------------- #
# plot the boxplot of the dispersal probability (at the last generation)

figure('disp_prob'+'_boxplot', figsize=(18, 11.25))
print( 'Creating figure:', 'disp_prob'+'_boxplot')

# remove NaN
tmpData = [x[~isnan(x)] for x in boxPlotDataDispProb[0][0]]
if compare == "Y":
    box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
    xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
    xticks(boxplotPositions, legends)
else:
    box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
    xticks(range(1,numTreatments+1), legends)
for patch, color in zip(box['boxes'], mycolors):
    patch.set_facecolor(color)
ylabel('Dispersal probability')
ylim(colRangesDisp)
savefig('disp_prob'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #
# plot the boxplot of the group stats (at the last generation)

for gs in range(numColGroupStats):
    figure(groupStatsFigLabels[gs]+'_boxplot', figsize=(18, 11.25))
    print ('Creating figure:', groupStatsFigLabels[gs]+'_boxplot')
    
    # remove NaN
    tmpData = [x[~isnan(x)] for x in boxPlotDataGroupStats[0][gs]]
    if compare == "Y":
        box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
        xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
        xticks(boxplotPositions, legends)
    else:
        box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
        xticks(range(1,numTreatments+1), legends)
    for patch, color in zip(box['boxes'], mycolors):
        patch.set_facecolor(color)
    ylabel(groupStatsYLabels[gs])
    savefig(groupStatsFigLabels[gs]+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #

# plot the boxplot for each column (at the last generation)
for i in range(numCol):
    # take only some columns
    if i not in dataIndexes:
        continue
    
    # -------------------------------------------------------------------------------------------- #
    
    figure(str(i)+'_separate'+'_boxplot', figsize=(18, 11.25))
    print( 'Creating figure:', str(i)+'_separate'+'_boxplot')
    
    if numCategories % 2 == 0:
        numRowCatFig=2
        numColCatFig=numCategories/2
    elif numCategories % 3 == 0:
        numRowCatFig=3
        numColCatFig=numCategories/3

    for l in range(numCategories):
        subplot(numRowCatFig,numColCatFig,l+1)
        # remove NaNq
        tmpData = [x[~isnan(x)] for x in boxPlotDataCategories[l][i]]
        if compare == "Y":
            box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
            xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
            locs, labels = xticks(boxplotPositions, legends)
        else:
            box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
            locs, labels = xticks(range(1,numTreatments+1), legends)
        for patch, color in zip(box['boxes'], mycolors):
            patch.set_facecolor(color)
        if colRanges[i][0] != colRanges[i][1]:
            ylim(colRanges[i])
        else:
            if minValues[i] != maxValues[i]:
                ylim([minValues[i],maxValues[i]])
        setp(labels, rotation=90)
        if (l+1) == 1 or (l % numColCatFig) == 0:
            ylabel(columns[i])
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
        title(categoryNames[l])
    savefig(str(i)+'_separate'+'_boxplot' +figureFormat)

    # -------------------------------------------------------------------------------------------- #
    
    figure(str(i)+'_separate2'+'_boxplot', figsize=(18, 11.25))
    print ('Creating figure:', str(i)+'_separate2'+'_boxplot')

    for n in range(numTreatments):
        subplot(numRowFig,numColFig,n+1)
        # remove NaN        
        tmpData = []
        for l in range(numCategories):
            tmpData += [boxPlotDataCategories[l][i][n]]
            
        tmpData = [x[~isnan(x)] for x in tmpData] 
        box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
        locs, labels = xticks(range(1,numCategories+1), categoryShort)
        #for patch in box['boxes']:
        #    patch.set_facecolor(mycolors[n])
        for patch, color in zip(box['boxes'], colors_categ):
            patch.set_facecolor(color)
        if colRanges[i][0] != colRanges[i][1]:
            ylim(colRanges[i])
        else:
            if minValues[i] != maxValues[i]:
                ylim([minValues[i],maxValues[i]])
        #setp(labels, rotation=90)
        if (n+1) == 1 or (n % numColFig) == 0:
            ylabel(columns[i])
        else:
            gca().yaxis.set_major_formatter(plt.NullFormatter())
        title(legends[n])
        
    plots = []
    for k in range(len(categoryNames)):
        plots += [plot([1,1], color=colors_categ[k], linewidth=2, label=categoryNames[k])]
        
    handles, labels = gca().get_legend_handles_labels()
    gca().legend(handles, labels, bbox_to_anchor=(0.5, 0.01), bbox_transform=gcf().transFigure, loc='lower center',
             numpoints=1, prop={'size':legendSize}, ncol=len(categoryNames), fancybox=True, shadow=True, borderaxespad=0.)
        
    for k in range(len(categoryNames)):
        plots[k][0].set_visible(False)
        
    savefig(str(i)+'_separate2'+'_boxplot' +figureFormat)

    # -------------------------------------------------------------------------------------------- #

    figure(str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    print ('Creating figure:', str(i)+'_aggregate'+'_boxplot')
    
    # remove NaN
    tmpData = [x[~isnan(x)] for x in boxPlotDataAggregate[0][i]]
    if compare == "Y":
        box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
        xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
        xticks(boxplotPositions, legends)
    else:
        box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
        xticks(range(1,numTreatments+1), legends)
    for patch, color in zip(box['boxes'], mycolors):
        patch.set_facecolor(color)
    if colRanges[i][0] != colRanges[i][1]:
        ylim(colRanges[i])
    ylabel(columns[i])
    savefig(str(i)+'_aggregate'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #
for i in range(numRel):
    figure('rel'+str(i)+'_aggregate'+'_boxplot', figsize=(18, 11.25))
    print( 'Creating figure:', 'rel'+str(i)+'_aggregate'+'_boxplot')
    
    # remove NaN
    tmpData = [x[~isnan(x)] for x in boxPlotDataRelatedness[0][i]]
    if compare == "Y":
        box = boxplot(tmpData, 0, flierSymbol, positions=boxplotPositions, widths = minDelta-0.2, patch_artist=True)
        xlim(boxplotPositions.min()-minDelta,boxplotPositions.max()+minDelta)
        xticks(boxplotPositions, legends, rotation=90)
    else:
        box = boxplot(tmpData, 0, flierSymbol, patch_artist=True)
        xticks(range(1,numTreatments+1), legends, rotation=90)
    for patch, color in zip(box['boxes'], mycolors):
        patch.set_facecolor(color)
    
    ylabel('Relatedness ' + '(' + str(i) + ')')
        
    savefig('rel'+str(i)+'_aggregate'+'_boxplot' +figureFormat)

# -------------------------------------------------------------------------------------------- #

# scatter plot and linear regression (at the last generation)

'''
figure('correlation_separate', figsize=(18, 11.25))
print 'Creating figure:', 'correlation_separate'

for n in range(numTreatments):
    ax=subplot(numRowFig,numColFig,n+1)
    for l in range(numCategories):
        xdata = boxPlotDataCategories[l][dataIndex1][n]
        ydata = boxPlotDataCategories[l][dataIndex2][n]
        
        # mask NaN
        xdata = np.ma.array(xdata, mask=np.isnan(xdata))
        xdata = np.ma.array(xdata, mask=np.isnan(ydata))
        ydata = np.ma.array(ydata, mask=np.isnan(xdata))
        ydata = np.ma.array(ydata, mask=np.isnan(ydata))

        line = plot(xdata, ydata, 'o', color=colors_categ[l], linewidth=2, label=categoryNames[l])
    
        if line is not None:
            mask = ~np.isnan(xdata) & ~np.isnan(ydata)
            if len(xdata[mask]) > 0 and len(ydata[mask]) > 0 and min(xdata[mask]) != max(xdata[mask]):
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata[mask], ydata[mask])
                    #print slope, intercept, r_value, p_value, std_err
                    #if not sys.version_info[:2] == (2,6):
                    #    note = 'r='+"{:g}".format(r_value)+'\n'+'p='+"{:g}".format(p_value)
                    #else:
                    #    note = 'r='+str(r_value)+'\n'+'p='+str(p_value)
                    #text(0.5, 0.75, note, bbox=dict(edgecolor='k'), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    plot(xdata[mask], [slope*x+intercept for x in xdata[mask]], linestyle=color=colors_categ[l], linewidth=2)
                except RuntimeWarning:
                    print ''
        line = None
    
    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        xlabel(columns[dataIndex1])
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel(columns[dataIndex2])
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())
    if colRanges[dataIndex1][0] != colRanges[dataIndex1][1]:
        xlim(colRanges[dataIndex1])
    if colRanges[dataIndex2][0] != colRanges[dataIndex2][1]:
        ylim(colRanges[dataIndex2])
    if numTreatments > 1:
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
        title(legends[n])
    savefig('correlation_separate'+figureFormat)
'''

'''
figure('correlation_aggregate_1', figsize=(18, 11.25))
print 'Creating figure:', 'correlation_aggregate_1'

for n in range(numTreatments):
    ax=subplot(numRowFig,numColFig,n+1)
    
    xdata = boxPlotDataAggregate[0][dataIndex1][n]
    ydata = boxPlotDataAggregate[0][dataIndex2][n]

    # mask NaN
    xdata = np.ma.array(xdata, mask=np.isnan(xdata))
    xdata = np.ma.array(xdata, mask=np.isnan(ydata))
    ydata = np.ma.array(ydata, mask=np.isnan(xdata))
    ydata = np.ma.array(ydata, mask=np.isnan(ydata))

    line = plot(xdata, ydata, '.', label=legends[n], color=mycolors[n])
    
    if line is not None:
        mask = ~np.isnan(xdata) & ~np.isnan(ydata)
        if len(xdata[mask]) > 0 and len(ydata[mask]) > 0 and min(xdata[mask]) != max(xdata[mask]):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(xdata[mask], ydata[mask])
                #print slope, intercept, r_value, p_value, std_err
                if not sys.version_info[:2] == (2,6):
                    note = 'r='+"{:g}".format(r_value)+'\n'+'p='+"{:g}".format(p_value)
                else:
                    note = 'r='+str(r_value)+'\n'+'p='+str(p_value)
                text(0.5, 0.75, note, bbox=dict(facecolor='w', edgecolor='k'), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                plot(xdata[mask], [slope*x+intercept for x in xdata[mask]],color='k',linewidth=1)
            except RuntimeWarning:
                print ''
    line = None

    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        xlabel(columns[dataIndex1])
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel(columns[dataIndex2])
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())
    if colRanges[dataIndex1][0] != colRanges[dataIndex1][1]:
        xlim(colRanges[dataIndex1])
    if colRanges[dataIndex2][0] != colRanges[dataIndex2][1]:
        ylim(colRanges[dataIndex2])
    if numTreatments > 1:
        title(legends[n])
    savefig('correlation_aggregate_1'+figureFormat)
    
figure('correlation_aggregate_2', figsize=(18, 11.25))
print 'Creating figure:', 'correlation_aggregate_2'

for n in range(numTreatments):
    ax=subplot(numRowFig,numColFig,n+1)
    
    #xdata = boxPlotDataAggregate[0][dataIndex1][n]
    xdata = boxPlotDataDispProb[0][0][n]
    ydata = boxPlotDataAggregate[0][dataIndex2][n]
    
    # mask NaN
    xdata = np.ma.array(xdata, mask=np.isnan(xdata))
    xdata = np.ma.array(xdata, mask=np.isnan(ydata))
    ydata = np.ma.array(ydata, mask=np.isnan(xdata))
    ydata = np.ma.array(ydata, mask=np.isnan(ydata))

    line = plot(xdata, ydata, '.', label=legends[n], color=mycolors[n])
    
    if line is not None:
        mask = ~np.isnan(xdata) & ~np.isnan(ydata)
        if len(xdata[mask]) > 0 and len(ydata[mask]) > 0 and min(xdata[mask]) != max(xdata[mask]):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(xdata[mask], ydata[mask])
                #print slope, intercept, r_value, p_value, std_err
                if not sys.version_info[:2] == (2,6):
                    note = 'r='+"{:g}".format(r_value)+'\n'+'p='+"{:g}".format(p_value)
                else:
                    note = 'r='+str(r_value)+'\n'+'p='+str(p_value)
                text(0.5, 0.75, note, bbox=dict(facecolor='w', edgecolor='k'), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                plot(xdata[mask], [slope*x+intercept for x in xdata[mask]],color='k',linewidth=1)
            except RuntimeWarning:
                print ''
    line = None

    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        #xlabel(columns[dataIndex1])
        xlabel('Dispersal probability')
    else:
        gca().xaxis.set_major_formatter(plt.NullFormatter())
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel(columns[dataIndex2])
    else:
        gca().yaxis.set_major_formatter(plt.NullFormatter())
    if colRanges[dataIndex1][0] != colRanges[dataIndex1][1]:
        xlim(colRangesDisp)
    if colRanges[dataIndex2][0] != colRanges[dataIndex2][1]:
        ylim(colRanges[dataIndex2])
    if numTreatments > 1:
        title(legends[n])
    savefig('correlation_aggregate_2'+figureFormat)
'''

# -------------------------------------------------------------------------------------------- #

for n in range(numTreatments):
    figure('correlation_aggregate_'+legends[n], figsize=(18, 11.25)) 
    print ('Creating figure:', 'correlation_aggregate_'+legends[n])
    f = 0
    
    for gs1 in range(numColStats):
        for gs2 in range(numColStats):
            #if True:
            #    ax=subplot(numColStats,numColStats,f+1)
            if gs1 > gs2:
                ax=subplot(numColStats-1,numColStats-1,f+1-numColStats-gs1+1)
                
                ax.locator_params(axis='both',tight=False,nbins=4)
                
                if gs2 == 0:
                    xdata = boxPlotDataGroupStats[0][1][n]
                elif gs2 == 1:
                    xdata = boxPlotDataGroupStats[0][3][n]
                elif gs2 == 2:
                    xdata = boxPlotDataAggregate[0][fitnessIndex][n]
                elif gs2 == 3:
                    xdata = boxPlotDataAggregate[0][altruismLevelIndex][n]
                elif gs2 == 4:
                    xdata = boxPlotDataDispProb[0][0][n]
                
                if gs1 == 0:
                    ydata = boxPlotDataGroupStats[0][1][n]
                elif gs1 == 1:
                    ydata = boxPlotDataGroupStats[0][3][n]
                elif gs1 == 2:
                    ydata = boxPlotDataAggregate[0][fitnessIndex][n]
                elif gs1 == 3:
                    ydata = boxPlotDataAggregate[0][altruismLevelIndex][n]
                elif gs1 == 4:
                    ydata = boxPlotDataDispProb[0][0][n]
                            
                # mask NaN
                xdata = np.ma.array(xdata, mask=np.isnan(xdata))
                xdata = np.ma.array(xdata, mask=np.isnan(ydata))
                ydata = np.ma.array(ydata, mask=np.isnan(xdata))
                ydata = np.ma.array(ydata, mask=np.isnan(ydata))
            
                line = plot(xdata, ydata, '.', label=legends[n], color=mycolors[n])
                
                if line is not None:
                    mask = ~np.isnan(xdata) & ~np.isnan(ydata)
                    if len(xdata[mask]) > 0 and len(ydata[mask]) > 0 and min(xdata[mask]) != max(xdata[mask]):
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(xdata[mask], ydata[mask])
                            #print slope, intercept, r_value, p_value, std_err
                            if not sys.version_info[:2] == (2,6):
                                note = 'r='+"{:g}".format(r_value)+'\n'+'p='+"{:g}".format(p_value)
                            else:
                                note = 'r='+str(r_value)+'\n'+'p='+str(p_value)
                            #text(0.5, 0.75, note, bbox=dict(facecolor='w', edgecolor='k'), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                            plot(xdata[mask], [slope*x+intercept for x in xdata[mask]],color='k',linewidth=1)
                        except RuntimeWarning:
                            print ('')
                line = None
                
                if (numColStats*numColStats-numColStats) < (f+1) and (f+1) <= (numColStats*numColStats):
                    xlabel(colStatsYLabels[gs2])
                else:
                    gca().xaxis.set_major_formatter(plt.NullFormatter())
                if f == 0 or (f % numColStats) == 0:
                    ylabel(colStatsYLabels[gs1])
                else:
                    gca().yaxis.set_major_formatter(plt.NullFormatter())
                
                if colCorrRanges[gs2][0] != colCorrRanges[gs2][1]:
                    xlim(colCorrRanges[gs2])
                else:
                    minXdata = min(xdata)
                    maxXdata = max(xdata)
                    #print minXdata, maxXdata, maxXdata-minXdata < 1
                    if maxXdata-minXdata < 1:
                        gca().set_autoscale_on(False)
                        xlim([(minXdata+maxXdata)/2-0.5,(minXdata+maxXdata)/2+0.5])
                if colCorrRanges[gs1][0] != colCorrRanges[gs1][1]:
                    ylim(colCorrRanges[gs1])
                else:
                    minYdata = min(ydata)
                    maxYdata = max(ydata)
                    #print minYdata, maxYdata, maxYdata-minYdata < 1
                    if maxYdata-minYdata < 1:
                        gca().set_autoscale_on(False)
                        ylim([(minYdata+maxYdata)/2-0.5,(minYdata+maxYdata)/2+0.5])
                savefig('correlation_aggregate_'+legends[n]+figureFormat)
                
            f = f+1

#show()
