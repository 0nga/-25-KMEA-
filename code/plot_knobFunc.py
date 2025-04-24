import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


plt.figure(figsize=(8,8))

x=np.arange(0,1,0.01)
#probabilità di fare male ad un padone
ped=0.8
#probabilità di fare male al passeggero
pas=1

y=x*(ped)/((pas)+(ped)*x)

# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('Prob. death pedestrian') 
# naming the y axis 
plt.ylabel('Knob Level') 
  
# giving a title to my graph 
plt.title('Knob Function') 
  
# function to show the plot 
plt.show() 