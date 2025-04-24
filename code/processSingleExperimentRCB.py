# -*- coding: utf-8 -*-

import sys
import glob
import math
import os
import cPickle as pickle

from results import *

import statsmodels.api as sm

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
for i in range(numFolders):
    if not os.listdir(folders[i]) == []:
        numRep = numRep+1

# number of generations for each repetition
numGenInRep = [0]*numRep
for i in range(numFolders):
    if not os.listdir(folders[i]) == []:
        dataFiles = sorted(glob.glob(folders[i] + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
        numGenInRep[i] = len(dataFiles)
numGen = np.max(numGenInRep)

# number of columns
tmpDataFiles = sorted(glob.glob(folders[0] + '/' + 'logs'  + '/' + '*' + '/' + 'gen*.txt'), key=natural_key)
data = readGenomeData(tmpDataFiles[0])
numLoci = len(data[0].split())
nRobots = len(data)

# debug messages
print "Nr. repetitions", numRep
print "Nr. generations", numGen

# nRobots x numLoci
if not useRealValues:
    print 'Error: this script can be executed only on real values'
    sys.exit()

# true if a constant (intercept) should be added to the regression model
useConstant = True

focalAltruism = np.empty(nRobots)
focalAltruism.fill(np.nan)

focalFitness = np.empty(nRobots)
focalFitness.fill(np.nan)

focalDispersal = np.empty(nRobots)
focalDispersal.fill(np.nan)

focalGenomes = np.empty((nRobots,numLoci))
focalGenomes.fill(np.nan)

partnersAltruism = np.empty(nRobots)
partnersAltruism.fill(np.nan)

partnersFitness = np.empty(nRobots)
partnersFitness.fill(np.nan)

partnersDispersal = np.empty(nRobots)
partnersDispersal.fill(np.nan)

partnersGenomes = np.empty((nRobots,numLoci))
partnersGenomes.fill(np.nan)

# NOTE: subtracting the avg. genome does not change the regression results
#popGenomes = np.empty(numLoci)
#popGenomes.fill(np.nan)

# -------------------------------------------------------------------------------------------- #

# for each repetition
for i in range(numFolders):
    # if the repetition folder is not empty
    if not os.listdir(folders[i]) == []:

        print "Processing folder", folders[i]
        # get the generations files
        filename = folders[i] + '/' + 'logs' + '/' + '*' + '/' + 'gen*.txt'
        dataFiles = sorted(glob.glob(filename), key=natural_key)
        
        file_regression_1 = open(folders[i] + '/' + 'logs' + '/' + 'regression_1.txt', 'w')
        file_regression_2 = open(folders[i] + '/' + 'logs' + '/' + 'regression_2.txt', 'w')
        file_regression_3 = open(folders[i] + '/' + 'logs' + '/' + 'regression_3.txt', 'w')
        file_regression_4 = open(folders[i] + '/' + 'logs' + '/' + 'regression_4.txt', 'w')
        file_regression_5 = open(folders[i] + '/' + 'logs' + '/' + 'regression_5.txt', 'w')
        file_regression_multi_gen = open(folders[i] + '/' + 'logs' + '/' + 'regression_multi_gen.txt', 'w')
        file_regression_multi_phen = open(folders[i] + '/' + 'logs' + '/' + 'regression_multi_phen.txt', 'w')
        
        # for each generation
        for gen in range(numGenInRep[i]):
            #if gen < numGenInRep[i]-1:
            #    continue
            
            j = gen
            #j = 0
            #print j
                        
            # read the j-th generation file
            data = readData(dataFiles[gen])
            positionData = readPositionData(dataFiles[gen])
            genomeData = readGenomeData(dataFiles[j])
                
            # for each robot
            for n in range(nRobots):
                genome_i = genomeData[n].split()
                # extract the genome of the n-th robot, in decimal format
                for k in range(numLoci):
                    if genome_i[k][0] == '+' or genome_i[k][0] == '-':
                        d = float(genome_i[k])
                    else:
                        d = scaleFromBits(int(genome_i[k]),nBits,geneMin,geneMax)
                    focalGenomes[n][k] = d
                
                focalAltruism[n] = data[n,altruismLevelIndex]
                focalFitness[n] = data[n,fitnessIndex]
                focalDispersal[n] = data[n,chambersIndex]
            
            #for k in range(numLoci):
            #    popGenomes[k]= np.mean(focalGenomes[:,k])
                        
            # create map of robots in groups
            groups = np.unique(positionData)
            robotsInGroups = {}
            for g in groups:
                robotsInGroups[g] = []
            for n in range(nRobots):
                g = positionData[n]
                robotsInGroups[g] += [n]
            
            # calculate average genetic value (weights) and phenotype (altruism) in partners
            for g in groups:
                for r1 in robotsInGroups[g]:
                    nPartners = len(robotsInGroups[g])-1
                    if nPartners > 0:
                        
                        for k in range(numLoci):
                            partnersGenomes[r1][k] = 0
                        partnersAltruism[r1] = 0
                        partnersFitness[r1] = 0
                        partnersDispersal[r1] = 0
                         
                        for r2 in robotsInGroups[g]:
                            if r1 != r2:
                                for k in range(numLoci):
                                    partnersGenomes[r1][k] += focalGenomes[r2][k]
                                partnersAltruism[r1] += focalAltruism[r2]
                                partnersFitness[r1] += focalFitness[r2]
                                partnersDispersal[r1] += focalDispersal[r2]
                        
                        for k in range(numLoci):
                            partnersGenomes[r1][k] /= nPartners
                        partnersAltruism[r1] /= nPartners
                        partnersFitness[r1] /= nPartners
                        partnersDispersal[r1] /= nPartners
                    else:
                        for k in range(numLoci):
                            partnersGenomes[r1][k] = np.nan
                        partnersAltruism[r1] = np.nan
                        partnersFitness[r1] = np.nan
                        partnersDispersal[r1] = np.nan
            
            mask = ~isnan(partnersFitness)
            partnersAltruism = partnersAltruism[mask]
            partnersFitness = partnersFitness[mask]
            partnersDispersal = partnersDispersal[mask]
            
            # relative fitness
            meanFitness = np.mean(focalFitness)
            f = focalFitness/meanFitness
            f1 = partnersFitness/meanFitness
            
            o = focalAltruism
            o1 = partnersAltruism    
            
            # -----------------------------------------------------------------
            # 1.
            
            # relatedness (genotype) cov(w,w1)/var(w), w=focal weight, w1=mean partners weight
            # eq. 2-6 Queller 2011, HRG Birch 2013
            
            r = np.empty(numLoci)
            for k in range(numLoci):
                w = focalGenomes[mask,k]
                w1 = partnersGenomes[mask,k]
                r[k] = np.cov(w,w1,ddof=0)[0][1]/np.var(w)
        
            # write relatedness data
            for r1 in r:
                file_regression_1.write(str(r1)+'\t')
            
            # c and b: regression of fitness on weights
            # f = alpha + beta(f,w)*w + beta(f,w1)*w1 + epsilon
            #
            # HRG: beta(f,w)+beta(f,w1)*cov(w,w1)/var(w) > 0
            #      -c + b*r > 0
            
            for k in range(numLoci):
                w = focalGenomes[mask,k]
                w1 = partnersGenomes[mask,k]
                X = np.column_stack((w, w1))
                    
                if useConstant:
                    X = sm.add_constant(X)
                
                model = sm.OLS(f, X)
                results = model.fit()
                
                for p in results.params:
                    file_regression_1.write(str(p)+'\t')
            
            # -----------------------------------------------------------------
            # 2.
            
            # relatedness (genotype-to-phenotype): cov(w,o1)/cov(w,o), w=focal weight, o=focal phenotype, o1=mean focal phenotype
            # eq. 7 Queller 2011, eq. 3.4 McGlothlin et al.
            
            r = np.empty(numLoci)
            for k in range(numLoci):
                w = focalGenomes[mask,k]
                r[k] = np.cov(w,o1,ddof=0)[0][1]/np.cov(w,o,ddof=0)[0][1]
                
            # write relatedness data
            for r1 in r:
                file_regression_2.write(str(r1)+'\t')
                
            X = np.column_stack((o, o1))
            
            if useConstant:
                X = sm.add_constant(X)
            
            model = sm.OLS(f, X)
            results = model.fit()
            
            for p in results.params:
                file_regression_2.write(str(p)+'\t')
            
            # -----------------------------------------------------------------
            # 3.
            
            # relatedness (genotype-to-phenotype): cov(w1,o)/cov(w,o), w=focal weight, o=focal phenotype, o1=mean focal phenotype
            # eq. 8 Queller 2011, inclusive fitness model
            
            r = np.empty(numLoci)
            for k in range(numLoci):
                w = focalGenomes[mask,k]
                w1 = partnersGenomes[mask,k]
                r[k] = np.cov(w1,o,ddof=0)[0][1]/np.cov(w,o,ddof=0)[0][1]
                
            # write relatedness data
            for r1 in r:
                file_regression_3.write(str(r1)+'\t')
            
            params = np.empty(2)
            
            X = np.column_stack((o, o1))
            
            if useConstant:
                X = sm.add_constant(X)
            
            # f = alpha + beta(f,o)*o + beta(f,o1)*o1 + epsilon
            model = sm.OLS(f, X)
            results = model.fit()
            
            params[0] = results.params[0]
            
            # f1 = alpha + beta(f1,o)*o + beta(f1,o1)*o1 + epsilon
            model = sm.OLS(f1, X)
            results = model.fit()
            
            params[1] = results.params[0]
            
            #params[0] = np.cov(f,o,ddof=0)[0][1]/np.var(o)
            #params[1] = np.cov(f1,o,ddof=0)[0][1]/np.var(o)
            
            for p in params:
                file_regression_3.write(str(p)+'\t')
            
            # -----------------------------------------------------------------
            # 4.
        
            # relatedness (phenotype): cov(o,o1)/var(o), o=focal phenotype, o1=mean focal phenotype
            # eq. 10.20 Rice book p.306
            
            r = np.cov(o,o1,ddof=0)[0][1]/np.var(o)
            file_regression_4.write(str(r1)+'\t')
            
            params = np.empty(2)
            
            params[0] = np.cov(f,o,ddof=0)[0][1]/np.var(o)
            params[1] = np.cov(f1,o1,ddof=0)[0][1]/np.var(o)
            
            for p in params:
                file_regression_4.write(str(p)+'\t')
            
            # -----------------------------------------------------------------
            # 5.
        
            # relatedness (phenotype): cov(o,o1)/var(o), o=focal phenotype, o1=mean focal phenotype
            # eq. 3.3 McGlothlin et al
            
            r = np.cov(o,o1,ddof=0)[0][1]/np.var(o)
            file_regression_5.write(str(r1)+'\t')
            
            # c and b: regression of fitness on phenotype
            # f = alpha + beta(f,o)*o + beta(f,o1)*o1 + epsilon
            #
            # HRG: beta(f,o)+beta(f,o1)*cov(w,o1)/cov(w,o) > 0
            #      -c + b*r > 0
            
            X = np.column_stack((o, o1))
            
            if useConstant:
                X = sm.add_constant(X)
                
            model = sm.OLS(f, X)
            results = model.fit()
            
            for p in results.params:
                file_regression_5.write(str(p)+'\t')
            
            # -----------------------------------------------------------------
            # 6a.
            
            # social vs non-social selection (multiple regression on weights)
            X = focalGenomes[mask,0]
            for k in range(1,numLoci):
                w = focalGenomes[mask,k]
                X = np.column_stack((X, w))
            for k in range(numLoci):
                w1 = partnersGenomes[mask,k]
                X = np.column_stack((X, w1))
            
            if useConstant:
                X = sm.add_constant(X)
            
            model = sm.OLS(f, X)
            results = model.fit()
            
            for p in results.params:
                file_regression_multi_gen.write(str(p)+'\t')
            
            # -----------------------------------------------------------------
            # 6b.
            
            # social vs non-social selection (multiple regression on traits)
            X = focalAltruism
            X = np.column_stack((X, focalDispersal))
            X = np.column_stack((X, partnersAltruism))
            X = np.column_stack((X, partnersDispersal))
            
            if useConstant:
                X = sm.add_constant(X)
            
            model = sm.OLS(f, X)
            results = model.fit()
            
            for p in results.params:
                file_regression_multi_phen.write(str(p)+'\t')
                
            # -----------------------------------------------------------------
            
            #print results.params
            #print results.mse_resid
            
            #print results.summary()
            #print results.mse_model
            #print results.mse_total
            #print results.ess
            #print results.resid
            #print results.rsquared
            
            #print sum((results.resid)**2)/(len(f)-2)
            #print results.resid**2-(f-(results.params[0]*X[:,0]+results.params[1]*X[:,1]))**2
            #print sum((f-(results.params[0]+results.params[1]*X[:,1]+results.params[2]*X[:,2]))**2)/(len(f)-2)
            #print sum((f-(results.params[0]*X[:,0]+results.params[1]*X[:,1]))**2)/(len(f)-2)
            #print np.cov(focalGenomes[mask,k],results.resid)[0][1]
            
            '''
            import matplotlib
            from pylab import *
            for k in range(numLoci):
                figure(str(k))
                plot(w,f,'.')
                figure(str(k)+"'")
                plot(w1,f,'.')
            show()
            '''    
            
            file_regression_1.write('\n')
            file_regression_2.write('\n')
            file_regression_3.write('\n')
            file_regression_4.write('\n')
            file_regression_5.write('\n')
            file_regression_multi_gen.write('\n')
            file_regression_multi_phen.write('\n')
        
        file_regression_1.close()
        file_regression_2.close()
        file_regression_3.close()
        file_regression_4.close()
        file_regression_5.close()
        file_regression_multi_gen.close()
        file_regression_multi_phen.close()