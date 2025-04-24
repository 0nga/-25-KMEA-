# -*- coding: utf-8 -*-

import sys
import glob
import math

import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.stats import *
from pylab import *

from results import *

# disable interactive mode (to create figures in background)
ioff()

# legend properties
legendSize = 14

# figures format
figureFormat='.pdf' #.svg .png .eps .pdf

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

    aggregate_trait = []
    aggregate_trait_tmp = []
    for tc in range(len(traitColumns)):
        aggregate_trait += [[]]
        aggregate_trait_tmp += [[]]

    print "Processing folder", folder
    rep_folders = sorted(glob.glob(folder + '/1*'), key=natural_key)

    r=1
    minDensity=np.inf
    maxDensity=-np.inf
    for rep_folder in rep_folders:
        
        print "Processing folder", rep_folder
        dataFiles = sorted(glob.glob(rep_folder + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
        if len(dataFiles) == 0:
            continue
        
        # take only the last generation
        for dataFile in [dataFiles[-1]]:
            data = readData(dataFile)
            positionBegin = readBirthChamberData(dataFile)
            positionEnd = readPositionData(dataFile)
            nrRobots = len(data)

            # for each robot
            for i in range(nrRobots):
            	if i == 0:
            	 	# initialize the traits structure
                    trait = []
                    for tc in range(len(traitColumns)):
                        trait += [[0]*nrRobots]
            
                # get the traits of interest
                for tc in range(len(traitColumns)):
                    trait[tc][i] = float(data[i][traitColumns[tc]])

            for tc in range(len(traitColumns)):
                aggregate_trait_tmp[tc] += [trait[tc]]

            fig=figure('robots-distribution-not-norm'+legends[n],figsize=(18, 11.25))
            fig.text(0.04, 0.5, 'Pop. density', va='center', rotation='vertical')
            options=folder.split('-')[1:]
            for option in options:
                if option.startswith('nrChambersPerRow'):
                    nrChambers = int(option[len('nrChambersPerRow'):])
            subplot(len(rep_folders),1,r)
            chambers=range(nrChambers)
            popDensityBegin=[0]*nrChambers
            popDensityEnd=[0]*nrChambers
            rowPositionBegin=[0]*nrRobots
            rowPositionEnd=[0]*nrRobots
            for i in range(nrRobots):
                rowPositionBegin[i]=int(positionBegin[i])
                rowPositionEnd[i]=int(positionEnd[i])
            rowPositionBegin=np.array(rowPositionBegin)
            rowPositionEnd=np.array(rowPositionEnd)
            for chamber in chambers:
                popDensityBegin[chamber] = len(data[rowPositionBegin==chamber])
                popDensityEnd[chamber] = len(data[rowPositionEnd==chamber])
            popDensityBegin=np.array(popDensityBegin)
            popDensityEnd=np.array(popDensityEnd)
            minD = min(popDensityBegin)
            maxD = max(popDensityBegin)
            if minD < minDensity:
                minDensity = minD
            if maxD > maxDensity:
                maxDensity = maxD
            minD = min(popDensityEnd)
            maxD = max(popDensityEnd)
            if minD < minDensity:
                minDensity = minD
            if maxD > maxDensity:
                maxDensity = maxD
            #normalize density
            #mind = min(min(popDensityBegin),min(popDensityEnd))
            #maxd = max(max(popDensityBegin),max(popDensityEnd))
            #popDensityBegin = (popDensityBegin-mind)/(maxd-mind)
            #popDensityEnd = (popDensityEnd-mind)/(maxd-mind)
            plot(chambers,popDensityBegin,'r',linewidth=2,label='Before')
            plot(chambers,popDensityEnd,'b',linewidth=2,label='After')
            if r == len(rep_folders):
                xlabel('Chambers')
            else:
                gca().xaxis.set_major_formatter(plt.NullFormatter())
            xlim([0,nrChambers-1])
            #ylim([0,1])
            savefig('robots-distribution-not-norm'+legends[n]+figureFormat)
        r+=1
            
    # adjust limits of repetitions plots
    fig=figure('robots-distribution-not-norm'+legends[n],figsize=(18, 11.25))
    for r in range(len(rep_folders)):
        ax = subplot(len(rep_folders),1,r+1)
        subplots_adjust(hspace = 0.3)
        ylim([minDensity,maxDensity])
        gca().locator_params(axis='y',tight=True,nbins=2)
        handles, labels = gca().get_legend_handles_labels()
        gca().legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=gcf().transFigure, loc='center right',
                     numpoints=1, prop={'size':legendSize}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)
    savefig('robots-distribution-not-norm'+legends[n]+figureFormat)
    
    nrRep=len(aggregate_trait_tmp[tc])
    for tc in range(len(traitColumns)):
        for i in range(nrRep):
            for j in range(nrRobots):
                aggregate_trait[tc] += [aggregate_trait_tmp[tc][i][j]]

    xdata=array(aggregate_trait[chambersIndex])
    ydata=array(aggregate_trait[altruismLevelIndex])
    zdata=array(aggregate_trait[fitnessIndex])
            
    figure('altruism-dispersal',figsize=(18, 11.25))
    subplot(numRowFig,numColFig,n+1)
    line = plot(xdata,ydata,'.')
    #xlim([0,1])
    ylim([0,1])

    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        xlabel('Dispersal probability')
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel('Altruism level')
    title(legends[n])
    savefig('altruism-dispersal'+figureFormat)

    fig=plt.figure('altruism-dispersal-fitness',figsize=(18, 11.25))
    ax=fig.add_subplot(numRowFig,numColFig,n+1,projection='3d')
    # normalize fitness
    minz = min(zdata)
    maxz = max(zdata)
    zdata = (zdata-minz)/(maxz-minz)
    #ax.scatter(xdata, ydata, zdata, marker='.')

    #ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])

    disc=0.05
    xdisc = np.arange(0, 1.1, disc)
    ydisc = np.arange(0, 1.1, disc)
    Z = np.zeros((len(xdisc),len(ydisc)))
    lz = np.zeros((len(xdisc),len(ydisc)))
    for i in range(nrRobots):
        if isnan(Z[int(xdata[i]/disc)][int(ydata[i]/disc)]):
            Z[int(xdata[i]/disc)][int(ydata[i]/disc)] = 0
            lz[int(xdata[i]/disc)][int(ydata[i]/disc)] = 0
        Z[int(xdata[i]/disc)][int(ydata[i]/disc)] += zdata[i]
        lz[int(xdata[i]/disc)][int(ydata[i]/disc)] += 1
    for i in range(len(xdisc)):
        for j in range(len(ydisc)):
            if lz[i][j] > 0:
                Z[i][j] = Z[i][j]/lz[i][j]
            if Z[i][j] > 1:
                Z[i][j] = 0
    X, Y = np.meshgrid(xdisc, ydisc)
    #surface
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=True) #linewidth=0,
    #wireframe
    #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    #contour plot
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.3)
    #cset = ax.contour(X, Y, Z, zdir='z', offset=-0.2, cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='x', offset=-0.2, cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='y', offset=1.2, cmap=cm.coolwarm)

    #ax.set_xlabel('Altruism level', labelpad=20)
    #ax.set_ylabel('Dispersal probability', labelpad=20)
    #ax.set_zlabel('Fitness', labelpad=20)

    start, end = gca().get_xlim()
    gca().xaxis.set_ticks(np.arange(start, end*1.01, (end-start)/4.0))
    start, end = gca().get_ylim()
    gca().yaxis.set_ticks(np.arange(start, end*1.01, (end-start)/4.0))
    start, end = gca().get_zlim()
    gca().zaxis.set_ticks(np.arange(start, end*1.01, (end-start)/4.0))

    title(legends[n])
    savefig('altruism-dispersal-fitness'+figureFormat)

    fig=plt.figure('altruism-dispersal-fitness-contour',figsize=(18, 11.25))
    ax=fig.add_subplot(numRowFig,numColFig,n+1)
    ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)

    if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
        xlabel('Dispersal probability')
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel('Altruism level')
    title(legends[n])
    savefig('altruism-dispersal-fitness-contour'+figureFormat)

    n += 1

# -------------------------------------------------------------------------------------------- #
#show()