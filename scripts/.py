import tif_reader as tif
import numpy as np
from scipy.integrate import trapz
import glob,os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.constants import e
from matplotlib.ticker import EngFormatter



charge = np.loadtxt("charges.txt")


#np.sum(a[a[:,1] == 3.])

def summing(simulation):
   
   chargearr = np.zeros(7)
   for sl in np.unique(simulation[:,1]):
      tmp = np.zeros(7)
      print len(np.unique(simulation[:,1]))
      print "slice number: ", sl
      same_slice = simulation[ simulation[:,1] == sl ]
      #print np.shape(same_slice)[0]
      number_of_arr = np.shape(same_slice)[0]
      #print "Number of array: ", number_of_arr
      #index = 0 
      for arr in same_slice:
         #print number_of_arr
         #print arr
         tmp = tmp + arr
      print "Number of arrays: ", number_of_arr
      print tmp
      tmp = tmp/np.array([number_of_arr, number_of_arr, 1., 1., 1., 1., 1. ])
      #print tmp
      chargearr = np.concatenate((chargearr,tmp))

      #print chargearr
   
   print len(np.unique( simulation[:,1]))
   return np.reshape(chargearr,(-1,7))[1:]
   #return chargearr
energy = []
for sim in range(500):
   simulation = charge[charge[:,0] == sim ]
 
   arr = summing(simulation)
   if len(arr) != 0:
      mask = arr < arr.max()
      index = np.where(mask==False)

      if index[0][0]-1 >= 0:
         i = index[0][0] -1
      else:
         i = 0
      j = index[0][0]+2

      if index[1][0]-1 >= 0:
         k = index[1][0]-1
      else:
         k=0
      l = index[1][0]+2
      energy.append(np.sum(arr))
   else:
      energy.append(0.0)

   #print arr[i:j,k:l]

   #print np.sum(arr)
   
plt.hist(energy, bins=50)
plt.grid(True)
plt.show()

#tmp = np.array([])
#for i in range(500):
   #simulation = charge[charge[:,0] == i]
   
   #print simulation[:,0]
   #print simulation[:,1]
   #simulation[simulation[:,1] == j]
   #simulation[simulation[:,1] == 3]
   #for j in range(5):
      #print simulation[:,0]
      #print j,simulation[simulation[:,1] == j]