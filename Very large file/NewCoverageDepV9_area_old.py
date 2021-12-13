# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:56:07 2020
Revised on Nov. 19
This file plots the particle deposition depending on the size of the particles in a loop.
Parameters:
    Details can be found in the INPUT PARAMETERS section

Notes:
    - the particle diameter is specified in mm in the txt file. Here we use microns
    - Figure saved as SVG
    - help on plotting found here https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html
    - rev 04:
        - title and figure labels are defined in the parameters section
        - axisOfInterest added (more flexible). Values: x is sagittal - y is rostrocaudal - z is height

@author: jmbouteiller
"""

#from numpy import genfromtxt
import pandas
import numpy as np
import matplotlib.pyplot as plt
from Coverage_Functions import total_area_Data_Refined
from Coverage_Functions import filter_Spray_RefinedV3
import os
from Coverage_Functions import total_area_filterData_Refined
from datetime import date
#===================== INPUT PARAMETERS ==========================
# inputFile = 'ang_dis_vel_1200000_part_viral_wide.txt'
inputFile = '3000_68.txt'
sprayFile ='vg_2.74_sweep_new.txt'
inputmesh = 'D:\Covid\Very large file\halfGeometry2.obj'
show3D = False
export3Dmesh = False
# Column names for the txt file
SprayIndex = np.array(['x', 'y', 'z','size', 'cone_angle','spray_vel'])
# SprayIndex = np.array(['x', 'y', 'z','size'])
# myIndex = np.array(['x', 'y', 'z','size','inhale_speed'])
myIndex = np.array(['x', 'y', 'z','size','dose'])
swabpercent = 1 #Coverage Percentage of using Swab 1 being 100%
swabpos = 1/4 # Swab coverage location in y axis, fraction of anterior coverage
Numsitcom = 4 #Number of situation to compare
threviral = 0.3 #threshold for viral particle coverage area infection
threspray = 0.3 #threshold for spray particle coverage area protection
Numofsize = 5 #Define how many columns of different sizes you want to show
# #===================== END INPUT PARAMETERS ==========================

raw_data = pandas.read_csv(inputFile, delim_whitespace=True, comment='%', header=0)
ttv = len(raw_data.columns) #number of columns in viral data
raw_data = raw_data.dropna()
raw_data = raw_data.to_numpy()
spray_data = pandas.read_csv(sprayFile, delim_whitespace=True, comment='%', header=0)
ttc = len(spray_data.columns) # number of columns in spray data
spray_data = spray_data.dropna()
spray_data = spray_data.to_numpy()
Npara = len(myIndex)
Npara2 = len(SprayIndex)
Numsit = int((ttc-1)/Npara2)
# Numsit = 1
NumsitV = int((ttv-1)/Npara)


##Different Viral situations
for j in range(NumsitV):
    #Initialize protection rate for spray, swab+spray
    Maxspray = 100
    Maxswab = 100
    MedSS = 0
    MaxD = pandas.DataFrame(data=raw_data[:, j * Npara + 1:(j + 1) * Npara + 1], columns=myIndex)
    sortdata = MaxD.sort_values(['size'])
    #Filter out viral particle at outlet
    viralout = sortdata[sortdata['z'] < (min(sortdata['z']) + max(sortdata['size']))]
    sortdata = sortdata[sortdata['z'] >= (min(sortdata['z']) + max(sortdata['size']))]
    coveragelen = min(sortdata['y']) + (max(sortdata['y']) - min(sortdata['y'])) * swabpos
    coveragelenx = -30
    NumofSit = int((np.shape(raw_data)[1] - 1) / Npara)
    sortdata['size'] = (sortdata['size'] * 100000 + .1).astype(int)  # +0.1 to make sure the it will not be rounded to 0
    size = np.array(sortdata['size'].unique())
    rangeS = int(len(size)/Numofsize)
    sizelist = np.array([size[0], size[rangeS], size[rangeS*2], size[rangeS*3], size[rangeS*4]])

    ##No Mask Analysis **
    depos = np.zeros(Numofsize)
    for i in range(0, Numofsize):
        tepdata = sortdata.loc[sortdata['size'] <= size[(i+1)*rangeS-1]]
        depos[i] = total_area_Data_Refined(inputmesh, tepdata, myIndex, False, False, threviral)

    ##Swab Analysis **
    deposswab = np.zeros(Numofsize)
    swabsort = sortdata.loc[(sortdata['y'] > coveragelen) | (sortdata['z'] > coveragelenx) | (sortdata['z'] < -50)] ## -30 and -50 are from COMSOL z axis
    for i in range(0,Numofsize):
        tepdata = swabsort.loc[swabsort['size'] <= size[(i+1)*rangeS-1]]
        deposswab[i] = total_area_Data_Refined(inputmesh, tepdata,myIndex, False, False,threviral)

    ##Mask Analysis **
    deposnew = np.zeros(Numofsize)
    partnew = sortdata.loc[sortdata['size'] == size[0]]
    # deposnew[0] = total_area_Data_Refined(inputmesh, partnew, myIndex, False, False, threviral)
    for i in range(1, len(size)):
        if size[i] < 40:
            tepdata = sortdata.loc[sortdata['size'] == size[i]]
            partnew = np.vstack((partnew, tepdata))
        elif size[i] < 110:
            tepdata = sortdata.loc[sortdata['size'] == size[i]]
            dropsize = int((0.25) * tepdata.shape[0])  ##Mask #1
            # dropsize = int(((size[i] - 1) /20 * 0.24 + 0.76) * tepdata.shape[0])  ##Mask #2
            # dropsize = int(((size[i] - 1)/20 * 0.70) * tepdata.shape[0]) ##Mask #3
            # dropsize = int(((size[i] - 1) /20 - 1) * -0.08 + 0.44) * tepdata.shape[0])  ##Mask #4
            drop_indices = np.random.choice(tepdata.index, dropsize, replace=False)
            tepdata = tepdata.drop(drop_indices)
            partnew = np.vstack((partnew, tepdata))
        else:
            tepdata = sortdata.loc[sortdata['size'] == size[i]]
            dropsize = int(((size[i] - 100) / 2000 * 0.50 + 0.25) * tepdata.shape[0])  ##Mask #1
            # dropsize = int(((size[i] - 1) /20 * 0.24 + 0.76) * tepdata.shape[0])  ##Mask #2
            # dropsize = int(((size[i] - 1)/20 * 0.70) * tepdata.shape[0]) ##Mask #3
            # dropsize = int(((size[i] - 1) /20 - 1) * -0.08 + 0.44) * tepdata.shape[0])  ##Mask #4
            drop_indices = np.random.choice(tepdata.index, dropsize, replace=False)
            tepdata = tepdata.drop(drop_indices)
            partnew = np.vstack((partnew, tepdata))
    partnew = pandas.DataFrame(data=partnew, columns=myIndex)
    for i in range(0,Numofsize):
        tepdata = partnew.loc[partnew['size'] <= size[(i + 1) * rangeS - 1]]
        deposnew[i] = total_area_Data_Refined(inputmesh, tepdata, myIndex, False, False, threviral)

    #Analyze Mask, Swab, No protection with one viral situation
    Max = total_area_Data_Refined(inputmesh, sortdata, myIndex, show3D, export3Dmesh,threviral)  # Total depos No Mask as control
    Maxswab = total_area_Data_Refined(inputmesh, swabsort, myIndex, show3D, export3Dmesh, threviral)  # Swab
    MaxM = total_area_Data_Refined(inputmesh, partnew, myIndex, show3D, export3Dmesh, threviral)  # Mask

    ## Different Spray situations
    sprayrange = np.zeros((0,3))
    swabrange = np.zeros((0,3))
    swabrange = swabrange + 100
    for k in range(Numsit):
        filename = '' + str(date.today().isoformat()) + sprayFile
        Spray = pandas.DataFrame(data = spray_data[:,k*Npara2+1:(k+1)*Npara2+1], columns = SprayIndex)
        # Filter out all the spray and viral particles at outlet,
        # +1 to ensure particle with all diameter will be covered
        sprayout = Spray[Spray['z'] < (min(Spray['z']) + max(Spray['size']))]
        Spray = Spray[Spray['z'] >= (min(Spray['z']) + max(Spray['size']))]
        SpID = filter_Spray_RefinedV3(Spray,sortdata,myIndex)

        #Spray Analysis
        tepspray = total_area_Data_Refined(inputmesh, SpID,myIndex, show3D, export3Dmesh,threviral) #Spray
        sprayrange = np.vstack([sprayrange,[tepspray, Spray.iloc[0]['cone_angle'], Spray.iloc[0]['spray_vel']]])
        if tepspray < Maxspray:
            Maxspray = tepspray
            Angle = Spray.iloc[0]['cone_angle']
            Velocity = Spray.iloc[0]['spray_vel']
        deposspray = np.zeros(Numofsize)
        for i in range(0,Numofsize):
            tepdata = SpID.loc[SpID['size'] <= size[(i+1)*rangeS-1]]
            deposspray[i] = total_area_Data_Refined(inputmesh, tepdata,myIndex, False, False,threviral)
        #Swab + Spray Analysis
        swabspray = np.zeros(Numofsize)
        sprayswab = SpID.loc[(SpID['y'] > coveragelen) | (SpID['z'] > coveragelenx)]
        for i in range(0,Numofsize):
            tepdata = sprayswab.loc[sprayswab['size'] <= size[(i+1)*rangeS-1]]
            swabspray[i] = total_area_Data_Refined(inputmesh, tepdata,myIndex, False, False,threviral)
        tepSS = total_area_Data_Refined(inputmesh, sprayswab, myIndex, show3D, export3Dmesh,threviral) #Swab+Spray
        swabrange = np.vstack([swabrange,[tepSS,Spray.iloc[0]['cone_angle'], Spray.iloc[0]['spray_vel']]])
        if tepSS < Maxswab:
            Maxss = tepSS
            AngleSS = Spray.iloc[0]['cone_angle']
            VelocitySS = Spray.iloc[0]['spray_vel']
            MedSS = Maxss
    spraystd = np.std(sprayrange)
    spraymean = np.mean(sprayrange)
    spraymin = np.min(sprayrange)
    swabmin = np.min(swabrange)


    barrange = np.array([spraymin,tepspray])
    swabbar = np.array([swabmin,tepSS])
    print('Best Cone angle is ' + str(Angle) + '\n' +'Best spray Velocity is '+ str(Velocity) + '\n' +
          'Best Cone angleSS is ' + str(AngleSS) + '\n' +'Best spray VelocitySS is '+ str(VelocitySS))







ax = plt.subplot(2,1,1)
ax.bar(sizelist/100-0.2, (depos - deposnew)/depos*100, width=0.2, color = 'k', ec = 'k', align='center')
ax.bar(sizelist/100, (depos - deposswab)/depos*100, width=0.2, color = 'darkgrey', ec = 'k', align='center')
ax.bar(sizelist/100+0.2, (depos - deposspray)/depos*100, width=0.2, color = 'w', ec = 'k', align='center')
legen = {'Mask':'k','Swab':'darkgrey', 'Spray':'w'}
labels = list(legen.keys())
handles = [plt.Rectangle((0,0),1,1, color=legen[label]) for label in labels]
ax.legend(handles,labels,prop = {"size":15})
ax.set_title('Protection rate with different situation of different size of particles',fontsize = 15)
plt.xlabel('Particle diameter(um)')
plt.ylabel('Protection Rate(%)')
xvalue = np.arange(Numsitcom)
# yvalue = [(Max - MaxM)/Max*100,(Max - Maxswab)/Max*100,(Max - Minswab)/Max*100, (Max - Maxspray)/Max*100,
#           (Max - MedSS)/Max*100, (Max - MedSM)/Max*100, (Max - Min)/Max*100]
yvalue = [(Max - MaxM)/Max*100,(Max - Maxswab)/Max*100, (Max - Maxspray)/Max*100,
          (Max - MedSS)/Max*100]
ax1 = plt.subplot(2,1,2)
ax1.bar(xvalue, yvalue,color = 'lightgrey', ec = 'k', align = 'center')
for index, value in enumerate(yvalue):
    ax1.text(index-0.1, value+1, str(int(value))+'%')
# xlabels = ['Mask', 'Swab', 'Swab+Mask', 'Spray', 'Spray+Swab', 'Spray+Mask', 'Spray+Swab+Mask']
xlabels = ['Mask', 'Swab', 'Spray', 'Spray+Swab']
plt.xticks(xvalue,xlabels)
ax1.set_title('Protection rate between different situations',fontsize = 15)
plt.xlabel('Protection Method')
plt.ylabel('Protection Rate(%)')
nameOfFig = str(int(spray_data[0,(k+1)*Npara-2]))+str(int(spray_data[0,(k+1)*Npara-1])) + filename
figure = plt.gcf()
figure.set_size_inches(11, 8)
plt.savefig(os.path.join(str(nameOfFig) + '.pdf'), dpi=300)
plt.savefig(os.path.join(str(nameOfFig) + '.svg'), format="svg")
plt.show()



with open("vg_2.7_results_spray", 'w') as outfile:

    for data_slice in sprayrange:  
        np.savetxt(outfile, data_slice, fmt='%i')
        
with open("vg_2.7_results_swab", 'w') as outfile:

    for data_slice in swabrange:  
        np.savetxt(outfile, data_slice, fmt='%i')


print("Finished!")






# fig.savefig( fileName+'.svg', dpi=150)