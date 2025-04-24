# -*- coding: utf-8 -*-

import sys
import glob
import math

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
figureFormat='.png' #.svg .png .eps .pdf

# save or show figures
saveFigures = True

# altruism threshold
altruismThreshold=float(sys.argv[1])

# result folder
folders = sys.argv[2:]

# data indexes to plot
dataIndexes = [altruismLevelIndex,
               fitnessIndex,
               ]

# number of treatments
numTreatments=len(folders)

# color map
NUM_COLORS = numTreatments
cm = plt.get_cmap(colorMap)
values=range(NUM_COLORS)
cNorm = colors.Normalize(vmin=values[0], vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

#mycolors = [cm(1.*n/NUM_COLORS) for n in values]    # manual scaling
#mycolors = [scalarMap.to_rgba(n) for n in values]   # map scaling

mycolors= ['red', 'blue', 'green']

# category styles for plots of number of individuals per category
style = ['r:', 'r', 'g:', 'g', 'b:', 'b']
width = [2, 2, 2, 2, 2, 2]

# number of categories
numCategories = len(categories)

# incremental index
n = 0

# -------------------------------------------------------------------------------------------- #

for folder in folders:
    print ("Processing folder", folder)
    
    if folder.endswith('/'):
        folder=folder[:-1]
    print(folder + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt')
    dataFiles = sorted(glob.glob(folder + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
    print(dataFiles)

    blockSize = 500

    data_i = readData(dataFiles[0])
    numGen = len(dataFiles)
    numRob = len(data_i)
    numCol = len(data_i[0])
    
    tmp = np.zeros(numRob)
    altruism = np.zeros(blockSize+1)
    dispersal = np.zeros(blockSize+1)
    meanNrRobots = np.zeros((numCategories,blockSize+1))
    
    genIndex = 0
    
    r = 0
    
    while genIndex < numGen:
        
        data = []
        positionData = []
        
        mean_ = np.array([])
        #min_ = np.array([])
        #max_ = np.array([])
        std_ = np.array([])
        
        repN = 0
        
        for thisGen in range(genIndex,min(genIndex+blockSize+1,numGen)):
            dataFile = dataFiles[thisGen]
            data_i = readData(dataFile)
            
            # remove values of altruism level corresponding to 0 collected food
            #for j in range(len(data_i)):
            #    if data_i[j][collectedFoodIndex] == 0:
            #        data_i[j][altruismLevelIndex] = np.nan
            data += [data_i]
            
            mean_i = nanmean(data_i,axis=0)
            #min_i = nanmin(data_i,axis=0)
            #max_i = nanmax(data_i,axis=0)
            std_i = nanstd(data_i,axis=0)
    
            mean_ = np.concatenate((mean_,mean_i),axis=0)
            #min_ = np.concatenate((min_,min_i),axis=0)
            #max_ = np.concatenate((max_,max_i),axis=0)
            std_ = np.concatenate((std_,std_i),axis=0)
        
        gen = arange(genIndex,min(genIndex+blockSize+1,numGen+1))
        
        mean_ = np.transpose(np.reshape(mean_,(len(mean_)/numCol,numCol)))
        #min_ = np.transpose(np.reshape(min_,(len(min_)/numCol,numCol)))
        #max_ = np.transpose(np.reshape(max_,(len(max_)/numCol,numCol)))
        std_ = np.transpose(np.reshape(std_,(len(std_)/numCol,numCol)))
        
        for i in range(numCol):
            
            # take only some columns
            if i not in dataIndexes:
                continue
    
            if i == altruismLevelIndex:
                for j in range(len(data)):
                    for k in range(len(data[j])):
                        tmp[k] = data[j][k][i]
                    tmp_nonan = tmp[~np.isnan(tmp)]
                    if len(tmp_nonan) != 0:
                        d = float(len(np.where(tmp_nonan>altruismThreshold)[0]))/len(tmp_nonan)
                    else:
                        d = 0
                    altruism[j] = d
                
                figure(numCol+1, figsize=(18, 11.25))
                plot(gen[0:len(mean_[i])], altruism[0:len(mean_[i])], linewidth=2, color=mycolors[n], label=folder.split('/')[-1])
                if len(folders) > 1:
                    handles, labels = gca().get_legend_handles_labels()
                    legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                       numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
                xlabel('Generations')
                ylabel('Altruists (%)')
                if saveFigures:
                    savefig(folder+'/'+'altruism'+figureFormat)
            
            figure(i, figsize=(18, 11.25))
            line = plot(gen[0:len(mean_[i])], mean_[i], linewidth=2, color=mycolors[n], label=folder.split('/')[-1])
            #NOTE: linewidth is needed to save figures correctly
            #fill_between(gen[0:len(mean_[i])], min_[i], max_[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            fill_between(gen[0:len(mean_[i])], mean_[i]-std_[i], mean_[i]+std_[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            #errorbar(gen[0:len(mean_[i])], mean_[i], yerr=std_[i], linestyle='-')
            #errorbar(gen[0:len(mean_[i])], (min+max)/2, yerr=[(min_[i]+max_[i])/2,(min_[i]+max_[i])/2], linestyle='-')
            #errorbar(gen[0:len(mean_[i])], mean_[i], yerr=[mean_[i]-min_[i],max_[i]-mean_[i]], linestyle='-')
            #legend(['mean ' + u'\u00B1' + ' std. dev.'],numpoints=1,loc=0)
            if len(folders) > 1:
                handles, labels = gca().get_legend_handles_labels()
                legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                   numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    
            xlabel('Generations')
            ylabel(columns[i])
            #gca().set_rasterized(True)
            if saveFigures:
                savefig(folder+'/'+str(i)+figureFormat)
            
            figure(numCol, figsize=(18, 11.25))
            nPlots=len(dataIndexes)
            ax = subplot(nPlots,1,repN+1)
            line = plot(gen[0:len(mean_[i])], mean_[i], linewidth=2, color=mycolors[n], label=folder.split('/')[-1])
            #NOTE: linewidth is needed to save figures correctly
            #fill_between(gen, min_[i], max_[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            fill_between(gen[0:len(mean_[i])], mean_[i]-std_[i], mean_[i]+std_[i], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            ylabel(columns[i], rotation='horizontal', labelpad=70)
            repN=repN+1
            if (repN < numCol):
                gca().xaxis.set_major_formatter(plt.NullFormatter())
            if (repN == numCol):
                xlabel('Generations')
                    
            if saveFigures:
                savefig(folder+'/'+'repetitions'+figureFormat)
    
        if saveFigures:
            savefig(folder+'/'+'repetitions'+figureFormat)
    
        genIndex = genIndex+blockSize
        
    n = n+1
    
# -------------------------------------------------------------------------------------------- #
#show()
