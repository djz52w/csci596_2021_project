# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:30:50 2021

@author: jinzedu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# =============================================================================
# Protection rate analysis
# =============================================================================

nameOfFig = "parameter_sweep"



"""Default values"""
no_protect = 7.722692314314461015
mask_protect = 6.702736777123953438
swab_protect = 6.387655654539832639

swab_minus = 0.7
"""Spray values"""
data = pd.read_csv('sprayrange.txt',sep=" ", header = None)
# data.columns = ["Situation 1", "Situation 2", "Situation 3", "Situation 4"]
print(data)


spray_only = data.loc[0,]

xvalue = np.arange(4)
std_value = (no_protect - min(spray_only))/no_protect*100 - (no_protect - max(spray_only))/no_protect*100
# yvalue = [(no_protect - MaxM)/no_protect*100,(no_protect - Maxswab)/no_protect*100,(no_protect - Minswab)/no_protect*100, (no_protect - Maxspray)/no_protect*100,
#           (no_protect - MedSS)/no_protect*100, (no_protect - MedSM)/no_protect*100, (no_protect - Min)/no_protect*100]
yvalue = [(no_protect - mask_protect)/no_protect*100,(no_protect - swab_protect)/no_protect*100, (no_protect - min(spray_only))/no_protect*100,
          (no_protect - (min(spray_only)-swab_minus))/no_protect*100]



fig1 = plt.figure(1)
stds   = [(0,0,0,0), [0,0,-std_value,-std_value]] # Standard deviation Data

plt.bar(xvalue, yvalue,yerr=stds, color = 'lightgrey', ec = 'k', align = 'center',capsize=10)






for index, value in enumerate(yvalue):
    plt.text(index-0.1, value+1, str(int(value))+'%')
# xlabels = ['Mask', 'Swab', 'Swab+Mask', 'Spray', 'Spray+Swab', 'Spray+Mask', 'Spray+Swab+Mask']
xlabels = ['Mask', 'Swab', 'Spray', 'Spray+Swab']
plt.xticks(xvalue,xlabels)
plt.title('Protection rate between different situations',fontsize = 15)
plt.xlabel('Protection Method')
plt.ylabel('Protection Rate(%)')

figure = plt.gcf()
figure.set_size_inches(11, 8)
plt.savefig(os.path.join(str(nameOfFig) + "1" + '.pdf'), dpi=300)
plt.savefig(os.path.join(str(nameOfFig) + "1" + '.svg'), format="svg")
plt.figure()
fig1.show()




fig2 = plt.figure(2)
protection_values = (no_protect - data.loc[0,])/no_protect * 100

plt.tricontourf(data.loc[3,], data.loc[2,], protection_values,cmap="jet")
plt.colorbar(orientation='vertical')

plt.title('Protection rates of different parameter combinations',fontsize = 15)
plt.xlabel('Spray velocity(m/s)')
plt.ylabel('Cone angle in rad')

figure = plt.gcf()
figure.set_size_inches(11, 8)
plt.savefig(os.path.join(str(nameOfFig) + "2" + '.pdf'), dpi=300)
plt.savefig(os.path.join(str(nameOfFig) + "2" + '.svg'), format="svg")
plt.figure()
fig1.show()


fig2.show()




