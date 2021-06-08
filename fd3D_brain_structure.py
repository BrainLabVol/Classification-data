import numpy as np
import pylab as pl
import sys
import time
import scipy.optimize
from sklearn import metrics
import math
import matplotlib.pyplot as plt
from scipy import stats
import statistics

def gather_voxels(volume,temp):
    '''
    Gather non zero voxels
    '''
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            if(temp in volume[i,j]):
            	for k in range(volume.shape[2]):
                        if (volume[i,j,k]==temp):
                            voxels.append((i,j,k))
    return voxels

def compute_hist( voxels,Gx,Lx,Gy,Ly,Gz,Lz,Ns,i):
    '''
    Compute Histogramm
    '''
    voxels[:,2]                       =           voxels[:,2] - min(voxels[:,2]) + Gx
    voxels[:,1]                       =           voxels[:,1] - min(voxels[:,1]) + Gy
    voxels[:,0]                       =           voxels[:,0] - min(voxels[:,0]) + Gz
    H, edges=np.histogramdd(voxels, bins=(np.arange(0,Lx+Gx,i),np.arange(0,Ly+Gy,i),np.arange(0,Lz+Gz,i)))
    Ns.append(np.sum(H>0))
    voxels[:,2]                       =           voxels[:,2] - min(voxels[:,2]) - Gx
    voxels[:,1]                       =           voxels[:,1] - min(voxels[:,1]) - Gy
    voxels[:,0]                       =           voxels[:,0] - min(voxels[:,0]) - Gz
    return Ns, H, edges

def initialize(voxels):

    #move the region to the center of the image
    x_length                          =           max(voxels[:,2]) - min(voxels[:,2])
    y_length                          =           max(voxels[:,1]) - min(voxels[:,1])
    z_length                          =           max(voxels[:,0]) - min(voxels[:,0])

    voxels[:,2]                       =           voxels[:,2] - min(voxels[:,2]) + x_length+1
    voxels[:,1]                       =           voxels[:,1] - min(voxels[:,1]) + y_length+1
    voxels[:,0]                       =           voxels[:,0] - min(voxels[:,0]) + z_length+1

    Lx                                =           (x_length)
    Ly                                =           (y_length)
    Lz                                =           (z_length)
    return x_length,y_length, z_length,Lx,Ly,Lz,voxels

def fractal_dimension(voxels,Lx,Ly,Lz,scale):
    '''
    Compute an optimized Df
    '''
    bestfit=[]
    for i in scale:
        bestfit,H,edges = compute_hist(voxels,0,Lx,0,Ly,0,Lz,bestfit,i)
    bestfit       =   np.asarray(bestfit)
    coeff = np.polyfit(np.log(scale), np.log(bestfit), 1)
    return coeff,bestfit

def lacunarity(voxels,Lx,Ly,Lz,x_length,y_length,z_length,s):
    '''
    Estimation of Fractal Dimension, lacunarity analysis plot and Prefactor lacunarity,
    '''
    #print(Ns)
    Ns  =         []
    Fractal_ar                    =           []
    Fractal_ar2                   =           []
    RMSE                          =           []
    minNs                         =           []
    A                             =           []
    Gx=0
    Gy=0
    Gz=0;
    offset=4

    if(x_length<2 or y_length<2 or z_length<2):
        offset=1
    elif(x_length<4 or y_length<4 or z_length<4):
        offset=2
    elif(x_length<8 or y_length<8 or z_length<8):
        offset=3

    if(offset<2):
        offset_x=x_length
        offset_y=y_length
        offset_z=z_length
        stepx=1
        stepy=1
        stepz=1
    else:
        offset_x=int((offset-1)*x_length/offset)
        offset_y=int((offset-1)*y_length/offset)
        offset_z=int((offset-1)*z_length/offset)
        stepx=int(x_length/offset)
        stepy=int(y_length/offset)
        stepz=int(z_length/offset)


    print("x length  ",x_length,y_length,z_length)
    print(stepx,stepy,stepz)
    if(offset==1):
        x_length

    while Gx < offset_x:
        while Gy < offset_y:
            while Gz < offset_z:
                #print(Lx,Ly,Lz)
                #print("field of view",Gx,Lx+Gx,Gy,Ly+Gy,Gz,Gz+Lz)
                for i in s:
                    Ns, H, edges           =       compute_hist(voxels,Gx,Lx,Gy,Ly,Gz,Lz,Ns,i)


                if(Ns.count(0)>0):
                    #print(Ns)
                    #print("NaN")
                    Ns=[]
                    break;
                else:
                    coeff                   =       np.polyfit(np.log(s), np.log(Ns), 1)
                    minNs.append(Ns)
                    D   =   -coeff[0]
                    coeff[0]=round(coeff[0],6)
                    coeff[1]=round(coeff[1],6)
                    Fractal_ar.append(-coeff[0])
                    Fractal_ar2.append(coeff[1])
                    #ax1.plot(np.log(s),np.log(Ns), '.',alpha=0.2)
                    Df                   =       -coeff[0]
                    coeff                =       np.polyfit(np.log(s**(-Df)),np.log(Ns), 1)
                    A.append(1/np.e**coeff[1])
                Ns=[]
                voxels[:,0]  =   voxels[:,0] + stepz
                Gz =  Gz + stepz
            voxels[:,0] = voxels[:,0] - Gz
            Gz =  0
            voxels[:,1]  =  voxels[:,1] + stepy
            Gy =  Gy + stepy
        voxels[:,1] =  voxels[:,1] - Gy
        Gy =  0
        voxels[:,2] = voxels[:,2] + stepx
        Gx =  Gx + stepx
    voxels[:,2] = voxels[:,2] - Gx

    if(len(A) == 0):
        PL=np.NaN
    else:
        A_mean =  sum(A[:])  /   len(A)
        print("A_mean ",A_mean)
        PL  =  sum(   (A[:]/A_mean - 1) ** 2) / len(A)
        print("Lacunarity is ",PL)



    return Fractal_ar,Fractal_ar2,PL,minNs,A


def function_compute( voxels,Lx,Ly,Lz):


    x_length, y_length, z_length, Lx, Ly, Lz, voxels =  initialize(voxels)

    voxels_brain_area                           =           len(voxels)
    print(voxels_brain_area)
    #total_voxels                                     =           (x_length + y_length + z_length)/3
    low_bound   =   np.log(voxels_brain_area)/10
    high_bound  =   np.log(voxels_brain_area)
    scale_lac                                         =           np.logspace(low_bound, np.log(high_bound), num=10, endpoint=True, base=np.e)
    scale_Df                                          =           np.logspace(0, np.log(3), num=40, endpoint=True, base=np.e)

    #print(scale_lac)
    #print(scale_Df)
    Fractal_ar, Fractal_ar2, PL, minNs,A  = lacunarity(voxels, Lx,Ly,Lz,x_length,y_length,z_length,scale_lac)
	

    if(len(Fractal_ar)>0):
        coeff, bestfit  = fractal_dimension(voxels,Lx,Ly,Lz,scale_Df)
        #ax1.plot(np.log(scale_lac),np.log(bestfit), '.',alpha=0.2)
        #ax1.plot(np.log(scale_lac), np.polyval(coeff,np.log(scale_Df)),color='mediumblue',alpha=0.8)
        #ax1.set_xlabel('ln s')
        #ax1.set_ylabel('ln N')
        #d="G = "+str(len(Fractal_ar))
        #ax1.set_title("Optimized Fractal Dimension is",-coeff[0])

        rmse                    =       np.sqrt(metrics.mean_squared_error(np.log(bestfit),np.log(scale_Df)*coeff[0]+coeff[1]))
        ax1.plot(np.log(scale_Df), np.polyval(coeff,np.log(scale_Df)),color='mediumblue',alpha=0.8)
        ax1.plot(np.log(scale_Df),np.log(bestfit), '.', mfc='red', alpha=0.2)
        d                       =       -round(coeff[0],4)
        d                       =       str(d)
        rmse                    =       round(rmse,4)
        rmse                    =       str(rmse)
        Df=-round(coeff[0],4)
        ax1.set_title("Df is "+str(Df)+" rmse = "+rmse)


        ax2.plot(Fractal_ar2,Fractal_ar, '.', mfc='blue',alpha=0.1)
        ax2.plot(coeff[1],-coeff[0], '.', mfc='red',alpha=0.8)
        ax2.set_xlabel('intercept')
        ax2.set_ylabel('Df')
        plt.savefig(str(sys.argv[2])+"/"+sys.argv[3]+"_"+sys.argv[5]+".pdf")

        fig2, (ax3, ax4) = plt.subplots(1, 2)
        fig2.tight_layout()
        ax3.plot(range(0,len(A),1),np.log(A), '.', mfc='blue',alpha=0.4)
        ax3.plot(len(A),PL, '.', mfc='red',alpha=0.8)
        #ax3.plot(coeff[1],-coeff[0], '.', mfc='red',alpha=0.8)
        ax3.set_xlabel('iterations')
        ax3.set_ylabel('A')
        my_dict = {'0': Fractal_ar}
        ax4.boxplot(my_dict.values())
        ax4.set_xticklabels(my_dict.keys())
        ax4.set_xlabel('Df')
        plt.savefig(str(sys.argv[2])+"/"+sys.argv[3]+"_"+sys.argv[5]+"2.pdf")
    else:
        coeff = np.array([np.NaN,np.NaN])
        rmse   =    10
    return -coeff[0] , PL, coeff[1], rmse


start_s                 =      time.time()
a                       =      str(sys.argv[1])
volume                  =      np.load(a)['vol']
temp                    =      int(sys.argv[3])
voxels=[]
voxels                  =      gather_voxels(volume,temp)
voxels                  =      pl.array(voxels)
voxels_brain_area       =      len(voxels)
total_voxels            =      volume.shape[2]*volume.shape[1]*volume.shape[0]
#print(voxels_brain_area)


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.tight_layout()
#calculate prefactor lacunarity
end_s1 = time.time()
#print("Seconds to calculate Fractal Dimension =", round( (end_s1-start_s) , 5) )

D,L,y_intercept, rmse = function_compute( voxels,volume.shape[2],volume.shape[1],volume.shape[0])


print("end")
import csv
#write fractal dimension of region to file
with open(sys.argv[4], 'a') as csv_file:
    s=str(sys.argv[5])+","+str(voxels_brain_area)+","+str(round(D, 6))
    csv_file.write(s+","+str(round(L,6))+","+str(round(y_intercept, 6))+","+rmse+"\n")

end_s = time.time()
print("Seconds to calculate Lacunarity =", round( (end_s-end_s1) , 5) )
print("Seconds since epoch =", round( (end_s-start_s) , 5) )	