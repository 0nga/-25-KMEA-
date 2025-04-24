# -*- coding: utf-8 -*-

import sys
import glob
import math
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')

from scipy.stats import *
from pylab import *

import matplotlib.colors as colors
import matplotlib.cm as cmx

from results import *

# disable interactive mode (to create figures in background)
ioff()

# legend properties
legendSize = 12

# figures format
figureFormat='.pdf' #.svg .png .eps .pdf

# compute standard deviation
computeStd = True

# choose what to compare (to create the plot legends)
# possible values: {"D", "Y", "L"}
compare = sys.argv[1]

# result folder
folders = sys.argv[2:]

traitColumns=[altruismLevelIndex,
              chambersIndex,
              fitnessIndex]

alpha=0.01

#print mpl.rcParams
mpl.rcParams['figure.max_open_warning']=maxFigures

n = 0

# number of treatments
numTreatments=len(folders)

# color map
NUM_COLORS = numTreatments
cm = plt.get_cmap(colorMap)
values = range(NUM_COLORS)
cNorm = colors.Normalize(vmin=values[0], vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

#mycolors = [cm(1.*v/NUM_COLORS) for c in values]    # manual scaling
mycolors = [scalarMap.to_rgba(v) for v in values]   # map scaling

# legends (one per treatment)
legends = ['']*numTreatments

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

for folder in folders:

    options=folder.split('-')[1:]
    # create legend
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
    
    print "Processing pickle file", pickleFileGenomes
    with open(folder + '/' + pickleFileGenomes, 'rb') as f:
        meanGenomes = pickle.load(f)

    numberRep = len(meanGenomes)

    for nrRep in range(numberRep):
        gen = 0
        # take all generations
        for j in range(len(meanGenomes[nrRep])):
            #if not (j == 0 or (j+1) % 250 == 0):
            #    continue
            
            nrGenes = len(meanGenomes[nrRep][0])
            nrGen = len(meanGenomes[nrRep])
            #nrGen = 5
            
            if gen == 0:
                genome = np.zeros([nrGenes,nrGen])
            
            for k in range(nrGenes):
                genome[k][gen] = meanGenomes[nrRep][j][k][1]
            
            gen = gen+1
        
        # -------------------------------------------------------------------------------------------- #
        print 'Creating figures:', 'gene-evolution'
        figure('gene-evolution' + legends[n], figsize=(18, 11.25))
        f = 0
        for l in range(nrGenes):
            for k in range(nrGenes):
                if l > k:
                    ax=subplot(nrGenes-1,nrGenes-1,f+1-nrGenes-l+1)
                    ax.locator_params(axis='both',tight=False,nbins=4)
                    
                    line = plot(genome[k],genome[l],'.-')
                    plot(genome[k][-1],genome[l][-1],'.',markersize=10)
                    
                    if (nrGenes*nrGenes-nrGenes) < (f+1) and (f+1) <= (nrGenes*nrGenes):
                        xlabel('Gene (' + str(k) + ')')
                    else:
                        gca().xaxis.set_major_formatter(plt.NullFormatter())
                    if f == 0 or (f % nrGenes) == 0:
                        ylabel('Gene (' + str(l) + ')')
                    else:
                        gca().yaxis.set_major_formatter(plt.NullFormatter())
                    
                    xlim([geneMin,geneMax])
                    ylim([geneMin,geneMax])
                
                f = f+1
        savefig('gene-evolution'+ legends[n]+figureFormat)

    # -------------------------------------------------------------------------------------------- #
    if n == 0:
        tmp = np.empty(numberRep)
        mean = np.empty((nrGenes,gen))
        if computeStd:
            std = np.empty((nrGenes,gen))
    
    generations = range(gen)
    
    print 'Creating figures:', 'gene-trend'
    figure('gene-trend', figsize=(18, 11.25))
    f = 1
    for k in range(nrGenes):
        ax=subplot(nrGenes,1,f)
        
        for g in range(nrGen):
            for nrRep in range(numberRep):
                tmp[nrRep] = meanGenomes[nrRep][g][k][1]
            mean[k][g] = np.mean(tmp)
            if computeStd:
                std[k][g] = np.std(tmp)
    
        if computeStd:
            line = plot(generations, mean[k], label=legends[n], color=mycolors[n], linewidth=2)
            fill_between(generations, mean[k]-std[k], mean[k]+std[k], facecolor=line[0].get_color(), edgecolor=line[0].get_color(), linewidth=0.0001, alpha=0.3)
            tmpMin = min(mean[k]-std[k])
            tmpMax = max(mean[k]+std[k])
        else:
            plot(generations, mean[k], label=legends[n], color=mycolors[n], linewidth=2)
                    
        ylabel('Gene (' + str(k) + ')')  
        if f == nrGenes:
            xlabel('Generations')
        else:
            gca().xaxis.set_major_formatter(plt.NullFormatter())
        
        #xlim([geneMin,geneMax])
        ylim([geneMin,geneMax])
        
        if numTreatments > 1:
            handles, labels = gca().get_legend_handles_labels()
            gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                        numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
        
        f = f+1

    savefig('gene-trend'+figureFormat)

    n += 1

# -------------------------------------------------------------------------------------------- #
#show()