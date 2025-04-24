import os
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon
import scipy.stats as stats


list_test=["nocost_noreward","yescost_noreward","nocost_yesreward","yescost_yesreward"]
predictions=pd.DataFrame()

print("Test whether cost or reward affect the accuracy. Null-hypotesis: they do not")
#for j in range(1,2):
for folder in list_test:
	path="/Users/aloreggia/Downloads/test/500ge/"+folder
	
	df=pd.read_csv(path+"/logs/0/detailed_acccuracy.txt",decimal=",",delimiter="\t")
	predictions[folder] = df.pred_y.astype(int)

#print(predictions)
threshold = 0.05
correction=True
stat, p = wilcoxon(predictions["nocost_noreward"], predictions["yescost_noreward"], correction=correction)

print("NONO-YESNO \t Stat %.3E \t mean p1 %.3E \t reject: %r" % (stat,p, p<threshold))
stat, p = wilcoxon(predictions["nocost_noreward"], predictions["nocost_yesreward"], correction=correction)

print("NONO-NOYES \t Stat %.3E \t mean p1 %.3E \t reject: %r" % (stat,p, p<threshold))

stat, p = wilcoxon(predictions["nocost_noreward"], predictions["yescost_yesreward"], correction=correction)

print("NONO-YESYES \t Stat %.3E \t mean p1 %.3E \t reject: %r" % (stat,p, p<threshold))
stat, p = wilcoxon(predictions["yescost_noreward"], predictions["yescost_yesreward"], correction=correction)

print("YESNO-YESYES \t Stat %.3E \t mean p1 %.3E \t reject: %r" % (stat,p, p<threshold))

f_value, p_value = stats.f_oneway(predictions["nocost_noreward"], predictions["yescost_noreward"])
print("OneWay %s \t mean p %.3E \t reject: %r" % ('value',p_value, p_value<threshold))
print("")

f_value, p_value = stats.kruskal(predictions["nocost_noreward"], predictions["yescost_noreward"])
print("Kruskal %s \t mean p %.3E \t reject: %r" % ('value',p_value, p_value<threshold))
print("")

