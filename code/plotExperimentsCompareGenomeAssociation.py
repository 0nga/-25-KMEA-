# -*- coding: utf-8 -*-

import sys
import glob
import math

import matplotlib
matplotlib.use('Agg')

from scipy.stats import *
from pylab import *

from results import *

# disable interactive mode (to create figures in background)
ioff()

# legend properties
legendSize = 12

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

plotGeneEvolution=True       # evolutionary trajectory in the genotype space
plotGeneTraitScatter=False   # scatter plot genes-traits at the last generation
plotGeneGeneScatter=False    # scatter plot genes-genes at the last generation

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

    aggregate_genome = []
    aggregate_genome_tmp = []

    aggregate_trait = []
    aggregate_trait_tmp = []
    for tc in range(len(traitColumns)):
        aggregate_trait += [[]]
        aggregate_trait_tmp += [[]]

    print "Processing folder", folder
    rep_folders = sorted(glob.glob(folder + '/1*'), key=natural_key)

    for rep_folder in rep_folders:
        
        print "Processing folder", rep_folder
        dataFiles = sorted(glob.glob(rep_folder + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
        if len(dataFiles) == 0:
            continue
        
        # -------------------------------------------------------------------------------------------- #
        if plotGeneEvolution:
            gen = 0
            # take all generations
            for j in range(len(dataFiles)):
                if not (j == 0 or (j+1) % 250 == 0):
                    continue
                
                genomeData = readGenomeData(dataFiles[j])
                
                nrRobots = len(genomeData)
                #nrGen = len(dataFiles[j])
                nrGen = 5
                
                # for each robot
                for i in range(nrRobots):
                    genome_i = genomeData[i].split()
                    
                    if i == 0 and gen == 0:
                        # initialize the genome structure
                        nrGenes = len(genome_i)
                        if useRealValues:
                            genome = np.zeros([nrGenes,nrGen])
                        else:
                            if genome_i[0][0] == '+' or genome_i[0][0] == '-':
                                print 'Error: the stored genome is real-valued'
                                sys.exit()
                            else:
                                genome = np.zeros([nrGenes*len(genome_i[0]),nrGen])
                                
                    # extract the genome of the i-th robot, in decimal format
                    for k in range(nrGenes):
                        if useRealValues:
                            if genome_i[k][0] == '+' or genome_i[k][0] == '-':
                                d = float(genome_i[k])
                            else:
                                d = scaleFromBits(int(genome_i[k]),nBits,geneMin,geneMax)
                            genome[k][gen] += d
                        else:
                            for b in range(nBits):
                                genome[k*nBits+b][gen] += float(genome_i[k][b])
            
                # average
                for k in range(nrGenes):
                    genome[k][gen] /= nrRobots
                    
                gen = gen+1
            
            print 'Creating figures:', 'gene-evolution'
            figure('gene-evolution' + legends[n], figsize=(18, 11.25))
            
            f = 0
            for l in range(nrGenes):
                for k in range(nrGenes):
                    if l > k:
                        ax=subplot(nrGenes-1,nrGenes-1,f+1-nrGenes-l+1)
                        ax.locator_params(axis='both',tight=False,nbins=4)
                        
                        line = plot(genome[k],genome[l],'.-')
                        #color = line[0].get_color()
                        #print k, l, f+1-nrGenes-l+1
                        
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
        # take only the last generation
        for dataFile in [dataFiles[-1]]:
            data = readData(dataFile)
            genomeData = readGenomeData(dataFile)
            nrRobots = len(genomeData)

            # for each robot
            for i in range(nrRobots):
                genome_i = genomeData[i].split()
                
                if i == 0:
                    # initialize the genome structure
                    nrGenes = len(genome_i)
                    if useRealValues:
                        # nrRobots x nrGenes
                        genome = [[0]*nrGenes for r in range(nrRobots)]
                    else:
                        if genome_i[0][0] == '+' or genome_i[0][0] == '-':
                            print 'Error: the stored genome is real-valued'
                            sys.exit()
                        else:
                            # nrRobots x nrGenes x nbits
                            genome = [[0]*nrGenes*len(genome_i[0]) for r in range(nrRobots)]
                    # initialize the traits structure
                    trait = []
                    for tc in range(len(traitColumns)):
                        trait += [[0]*nrRobots]
        
                # get the traits of interest
                for tc in range(len(traitColumns)):
                    trait[tc][i] = float(data[i][traitColumns[tc]])
                
                # extract the genome of the i-th robot, in decimal format
                for k in range(nrGenes):
                    if useRealValues:
                        if genome_i[k][0] == '+' or genome_i[k][0] == '-':
                            d = float(genome_i[k])
                        else:
                            d = scaleFromBits(int(genome_i[k]),nBits,geneMin,geneMax)
                            #print toBitString(int(genome_i[k]), nBits) + "\t" + str(d)
                        genome[i][k] = d
                    else:
                        for b in range(nBits):
                            genome[i][k*nBits+b]=float(genome_i[k][b])
                    
            aggregate_genome_tmp += [genome]
            for tc in range(len(traitColumns)):
                aggregate_trait_tmp[tc] += [trait[tc]]

    # -------------------------------------------------------------------------------------------- #

    nrRep=len(aggregate_trait_tmp[tc])
    
    for i in range(nrRep):
        for j in range(nrRobots):
            aggregate_genome += [aggregate_genome_tmp[i][j]]

    for tc in range(len(traitColumns)):
        for i in range(nrRep):
            for j in range(nrRobots):
                aggregate_trait[tc] += [aggregate_trait_tmp[tc][i][j]]

    aggregate_genome = array(aggregate_genome)
    aggregate_genome = np.transpose(aggregate_genome)

    nrLoci=len(aggregate_genome)
    genes=range(nrLoci)

    # -------------------------------------------------------------------------------------------- #
    # plot the gene-trait correlations
    
    R_genes_traits=[]
    p_genes_traits=[]
    
    for tc in range(len(traitColumns)):
        
        r_values=[0]*nrLoci
        p_values=[0]*nrLoci

        for k in range(nrLoci):
            
            xdata=array(aggregate_genome[k])
            ydata=array(aggregate_trait[tc])

            mask = ~np.isnan(xdata) & ~np.isnan(ydata)
            xdata = xdata[mask]
            ydata = ydata[mask]

            slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
            #print slope, intercept, r_value, p_value, std_err
            #r_value,p_value = spearmanr(xdata,ydata,axis=0)
            #r_value,p_value = pearsonr(xdata,ydata)
            #r_value,p_value = kendalltau(xdata,ydata)
            #print r_value
            #print p_value
            
            r_values[k] = r_value
            p_values[k] = p_value
            if p_values[k] > alpha:
                r_values[k] = 0.0
            
            if plotGeneTraitScatter:
                print 'Creating figure:', ('gene' + '_' + str(k) + ' vs ' + columns[traitColumns[tc]] + '_' + legends[n])
                figure('gene' + '_' + str(k) + ' vs ' + columns[traitColumns[tc]] + '_' + legends[n])
                line = plot(aggregate_genome[k],aggregate_trait[tc],'.')
                color = line[0].get_color()
                #x = np.linspace(geneMin, geneMax, 50)
                #y = [slope*x_+intercept for x_ in x]
                #plot(x, y, color, lw=2)
                xlim([geneMin,geneMax])
                if colRanges[traitColumns[tc]][0] != colRanges[traitColumns[tc]][1]:
                    ylim(colRanges[traitColumns[tc]])
                xlabel('Gene (' + str(k) + ')')
                ylabel(columns[traitColumns[tc]])
                savefig('gene' + '_' + str(k) + ' vs ' + columns[traitColumns[tc]] + '_' + legends[n]+figureFormat)
        
        p_values=array(p_values)
        mask_p_values=p_values>alpha
        r_values_significant=np.ma.array(r_values,mask=mask_p_values)
        r_values_not_significant=np.ma.array(r_values,mask=~mask_p_values)
        genes_significant=np.ma.array(genes,mask=mask_p_values)
        genes_not_significant=np.ma.array(genes,mask=~mask_p_values)

        r_values_significant=ma.filled(r_values_significant,np.nan)
        r_values_not_significant=ma.filled(r_values_not_significant,np.nan)

        R_genes_traits += [r_values]
        p_genes_traits += [p_values]
        
        '''
        figure(columns[traitColumns[tc]],figsize=(18, 11.25))
        subplot(numRowFig,numColFig,n+1)
        axhline(y=0,ls='-',color='k')

        xlim([genes[0]-1,genes[-1]+1])

        plot(r_values,'k.')
        #plot(r_values, 'k', marker=r"${}$".format('(**)'), markersize=16)
        #vlines(genes, [0], r_values, linestyles='dashed')

        #plot(r_values_significant,'k*')
        vlines(genes_significant, 0, r_values_significant, linestyle='solid')
        vlines(genes_not_significant, 0, r_values_not_significant, linestyle='dashed')
        ylim([-1,1])
        
        grid()

        if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
            xlabel('Gene index')
        if (n+1) == 1 or (n % numColFig) == 0:
            #ylabel('Correlation (-Log(p_genes_traits))')
            ylabel('Correlation')
        title(legends[n])
        savefig(columns[traitColumns[tc]]+figureFormat)
        '''

    # -------------------------------------------------------------------------------------------- #
    # plot the gene-gene correlations
    
    R_genes_genes=[]
    p_genes_genes=[]
    
    for l in range(nrLoci):
    
        r_values=[0]*nrLoci
        p_values=[0]*nrLoci
        
        for k in range(nrLoci):
            
            xdata=array(aggregate_genome[k])
            ydata=array(aggregate_genome[l])
            
            mask = ~np.isnan(xdata) & ~np.isnan(ydata)
            xdata = xdata[mask]
            ydata = ydata[mask]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(xdata,ydata)
            #print slope, intercept, r_value, p_value, std_err
            #r_value,p_value = spearmanr(xdata,ydata,axis=0)
            #r_value,p_value = pearsonr(xdata,ydata)
            #r_value,p_value = kendalltau(xdata,ydata)
            #print r_value
            #print p_value
            
            r_values[k] = r_value
            p_values[k] = p_value
            if p_values[k] > alpha:
                r_values[k] = 0.0
                
            if plotGeneGeneScatter:
                print 'Creating figure:', ('gene' + '_' + str(k) + ' vs ' + 'gene' + '_' + str(l) + '_' + legends[n])
                fig = figure('gene' + '_' + str(k) + ' vs ' + 'gene' + '_' + str(l) + '_' + legends[n])
                line = plot(aggregate_genome[k],aggregate_genome[l],'.')
                color = line[0].get_color()
                #x = np.linspace(geneMin, geneMax, 50)
                #y = [slope*x_+intercept for x_ in x]
                #plot(x, y, color, lw=2)
                xlim([geneMin,geneMax])
                ylim([geneMin,geneMax])
                xlabel('Gene (' + str(k) + ')')
                ylabel('Gene (' + str(l) + ')')
                savefig('gene' + '_' + str(k) + ' vs ' + 'gene' + '_' + str(l) + '_' + legends[n]+figureFormat)

        p_values=array(p_values)
        mask_p_values=p_values>alpha
        r_values_significant=np.ma.array(r_values,mask=mask_p_values)
        r_values_not_significant=np.ma.array(r_values,mask=~mask_p_values)
        genes_significant=np.ma.array(genes,mask=mask_p_values)
        genes_not_significant=np.ma.array(genes,mask=~mask_p_values)
        
        r_values_significant=ma.filled(r_values_significant,np.nan)
        r_values_not_significant=ma.filled(r_values_not_significant,np.nan)
        
        R_genes_genes += [r_values]
        p_genes_genes += [p_values]

        '''
        figure('gene'+'_'+str(l),figsize=(18, 11.25))
        subplot(numRowFig,numColFig,n+1)
        axhline(y=0,ls='-',color='k')

        xlim([genes[0]-1,genes[-1]+1])

        plot(r_values,'k.')
        #plot(r_values, 'k', marker=r"${}$".format('(**)'), markersize=16)
        #vlines(genes, [0], r_values, linestyles='dashed')

        #plot(r_values_significant,'k*')
        vlines(genes_significant, 0, r_values_significant, linestyle='solid')
        vlines(genes_not_significant, 0, r_values_not_significant, linestyle='dashed')
        ylim([-1,1])
    
        grid()

        if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
            xlabel('Gene index')
        if (n+1) == 1 or (n % numColFig) == 0:
            #ylabel('Correlation (-Log(p_genes_traits))')
            ylabel('Correlation')
        title(legends[n])
        savefig('gene'+'_'+str(l)+figureFormat)
        '''
    
    # -------------------------------------------------------------------------------------------- #
    print 'Creating figures:', 'correlation maps'
    
    # genes-traits correlation maps
    R_genes_traits=array(R_genes_traits)
    figure('corrcoef_genes_traits',figsize=(18, 11.25))
    subplot(numRowFig,numColFig,n+1)
    #axis('scaled')
    pcolor(R_genes_traits,vmin=-1,vmax=1,edgecolors='k',cmap=cm.bwr) #cm.coolwarm
    #if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
    xlabel('Gene index')
    colorbar(orientation='horizontal',ticks=[-1, 0, 1], fraction=0.046, pad=0.2)
    xticks(arange(0.5,nrLoci+1),range(0,nrLoci))
    columNames = [columns[t] for t in traitColumns]
    if (n+1) == 1 or (n % numColFig) == 0:
        yticks(arange(0.5,len(traitColumns)+1),columNames)
    else:
        yticks([],[])
    tick_params(axis='both', length=0, width=0)
    xlim(0,nrLoci)
    ylim(0,len(traitColumns))
    title(legends[n])
    savefig('corrcoef_genes_traits'+figureFormat)

    # genes-genes correlation maps
    R_genes_genes=array(R_genes_genes)
    figure('corrcoef_genes_genes',figsize=(18, 11.25))
    subplot(numRowFig,numColFig,n+1)
    #axis('scaled')
    pcolor(R_genes_genes,vmin=-1,vmax=1,edgecolors='k',cmap=cm.bwr) #cm.coolwarm
    #if numColFig*numRowFig-numColFig < (n+1) and (n+1) <= numColFig*numRowFig:
    xlabel('Gene index')
    colorbar(orientation='horizontal',ticks=[-1, 0, 1], fraction=0.046, pad=0.2)
    xticks(arange(0.5,nrLoci+1),range(0,nrLoci))
    if (n+1) == 1 or (n % numColFig) == 0:
        ylabel('Gene index')
        yticks(arange(0.5,nrLoci+1),range(0,nrLoci))
        #yticks(arange(0.5,len(traitColumns)+1),columNames)
    else:
        yticks([],[])
    tick_params(axis='both', length=0, width=0)
    xlim(0,nrLoci)
    ylim(0,nrLoci)
    title(legends[n])
    savefig('corrcoef_genes_genes'+figureFormat)
    # -------------------------------------------------------------------------------------------- #

    n += 1

# -------------------------------------------------------------------------------------------- #
#show()