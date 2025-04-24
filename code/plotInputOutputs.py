# -*- coding: utf-8 -*-

import sys
import glob
import math
import os
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

from mpl_toolkits.mplot3d import Axes3D

from results import *

def readNetIOData(filename):
    #data = np.genfromtxt(filename,delimiter=" ",skip_header=1)
    data = np.loadtxt(filename,delimiter=" ",skiprows=1)
    return data

# disable interactive mode (to create figures in background)
ioff()

# legend properties
legendSize = 14

# figures format
figureFormat='.pdf' #.svg .png .eps .pdf

# choose what to compare (to create the plot legends)
# possible values: {"D", "Y", "L", "SelectionTypes", "DensityInputs", "Inputs", "ZeroInputs", "AltrNonAltr", "n", "t"}
compare = sys.argv[1]

# result folder
folders = sys.argv[2:]

# number of treatments
numTreatments=len(folders)

# legends (one per treatment)
legends = ['']*numTreatments

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

    print ("Processing folder", folder)

    repetitions = sorted(glob.glob(folder + '/1*'), key=natural_key)

    data = {}
    for repetition in repetitions:
        # get the input/output file
        filename = repetition + '/' + 'altruistic_acts.txt'
        data[repetition] = readNetIOData(filename)
        lenData = len(data[repetition])
        # comment this to use the full data (2 generations)
        #data[repetition] = data[repetition][9*lenData/10:,:]
        data[repetition] = data[repetition][lenData/2:,:]

    numRep = len(repetitions)
    concatData = data[repetitions[0]]
    for i in range(1,numRep):
        concatData = np.concatenate((concatData, data[repetitions[i]]), axis=0)

    # plot the trends for each column along the generations
    if n == 0:
        fig_dispersal = figure('Dispersal', figsize=(18, 11.25))
        ax_dispersal = fig_dispersal.add_subplot(221, projection='3d')
        ax_xy1 = fig_dispersal.add_subplot(222)
        ax_xz1 = fig_dispersal.add_subplot(223)
        ax_yz1 = fig_dispersal.add_subplot(224)
        
        ax_dispersal.set_xlabel('Disp Dist')
        ax_dispersal.set_ylabel('Pop Dens')
        ax_dispersal.set_zlabel('Dispersal')
        ax_dispersal.set_xlim([-1,1])
        ax_dispersal.set_ylim([-1,1])
        ax_dispersal.set_zlim([0,1])

        ax_xy1.set_xlabel('Disp Dist')
        ax_xy1.set_ylabel('Pop Dens')
        ax_xy1.set_xlim([-1,1])
        ax_xy1.set_ylim([-1,1])
        
        ax_xz1.set_xlabel('Disp Dist')
        ax_xz1.set_ylabel('Dispersal')
        ax_xz1.set_xlim([-1,1])
        ax_xz1.set_ylim([0,1])
        
        ax_yz1.set_xlabel('Pop Dens')
        ax_yz1.set_ylabel('Dispersal')
        ax_yz1.set_xlim([-1,1])
        ax_yz1.set_ylim([0,1])

    ax_dispersal.plot(concatData[:,1], concatData[:,2], concatData[:,3], '.', c=mycolors[n], label=legends[n])
    ax_xy1.plot(concatData[:,1], concatData[:,2], '.', c=mycolors[n], label=legends[n])
    ax_xz1.plot(concatData[:,1], concatData[:,3], '.', c=mycolors[n], label=legends[n])
    ax_yz1.plot(concatData[:,2], concatData[:,3], '.', c=mycolors[n], label=legends[n])

    if numTreatments > 1:
        handles, labels = ax_dispersal.get_legend_handles_labels()
        ax_dispersal.legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=fig_dispersal.transFigure, loc='center right',
                     numpoints=1, prop={'size':8}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)

    fig_dispersal.savefig('input_output_map_dispersal'+figureFormat)

    if n == 0:
        fig_altruism = figure('Altruism', figsize=(18, 11.25))
        ax_altruism = fig_altruism.add_subplot(221, projection='3d')
        ax_xy2 = fig_altruism.add_subplot(222)
        ax_xz2 = fig_altruism.add_subplot(223)
        ax_yz2 = fig_altruism.add_subplot(224)
        
        ax_altruism.set_xlabel('Disp Dist')
        ax_altruism.set_ylabel('Pop Dens')
        ax_altruism.set_zlabel('Altruism')
        ax_altruism.set_xlim([-1,1])
        ax_altruism.set_ylim([-1,1])
        ax_altruism.set_zlim([0,1])

        ax_xy2.set_xlabel('Disp Dist')
        ax_xy2.set_ylabel('Pop Dens')
        ax_xy2.set_xlim([-1,1])
        ax_xy2.set_ylim([-1,1])
        
        ax_xz2.set_xlabel('Disp Dist')
        ax_xz2.set_ylabel('Altruism')
        ax_xz2.set_xlim([-1,1])
        ax_xz2.set_ylim([0,1])
        
        ax_yz2.set_xlabel('Pop Dens')
        ax_yz2.set_ylabel('Altruism')
        ax_yz2.set_xlim([-1,1])
        ax_yz2.set_ylim([0,1])

    ax_altruism.plot(concatData[:,1], concatData[:,2], concatData[:,4], '.', c=mycolors[n], label=legends[n])
    ax_xy2.plot(concatData[:,1], concatData[:,2], '.', c=mycolors[n], label=legends[n])
    ax_xz2.plot(concatData[:,1], concatData[:,4], '.', c=mycolors[n], label=legends[n])
    ax_yz2.plot(concatData[:,2], concatData[:,4], '.', c=mycolors[n], label=legends[n])

    if numTreatments > 1:
        handles, labels = ax_altruism.get_legend_handles_labels()
        ax_altruism.legend(handles, labels, bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=fig_altruism.transFigure, loc='center right',
                 numpoints=1, prop={'size':8}, ncol=1, fancybox=True, shadow=True, borderaxespad=0.)

    fig_altruism.savefig('input_output_map_altruism'+figureFormat)

    n = n+1

# -------------------------------------------------------------------------------------------- #
#show()