#import seaborn as sns
import style
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import rc
import bw

# Read silicon and CdTe cross sections and plot


fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(111)
si = np.loadtxt("data/si_photon_crossec.dat", skiprows=6, usecols=(0,1,2,3,4,5,6,7))
data = np.loadtxt("data/cdte_photon_crossec.dat", skiprows=6, usecols=(0,1,2,3,4,5,6,7))

colors = style.color_palette(style.cold_hot, 8)
#ax.loglog(data[:,0], data[:,1],'y', label=r'$\sigma_{raleigh}$')
ax.loglog(si[:,0], si[:,2],color=colors[0], label=r'Si $ \sigma_{compton}$')
ax.loglog(si[:,0], si[:,3],color=colors[1], label=r'Si $\sigma_{photoelectric}$')
#line.set_dashes([8, 4, 2, 4, 2, 4])
ax.loglog(si[:,0], si[:,4]+si[:,5],color=colors[2], label=r'Si $\sigma_{pair\,production}$')
#ax.loglog(data[:,0], data[:,5],'m', label='')
ax.loglog(si[:,0], si[:,6],color=colors[3], label=r'Si $\sigma_{total}$')

#ax.loglog(data[:,0], data[:,1],'y', label=r'$\sigma_{raleigh}$')
ax.loglog(data[:,0], data[:,2],color=colors[4], label=r'CdTe $\sigma_{compton}$')
ax.loglog(data[:,0], data[:,3],color=colors[5], label=r'CdTe $\sigma_{photoelectric}$')
#line.set_dashes([8, 4, 2, 4, 2, 4])
ax.loglog(data[:,0], data[:,4]+data[:,5],color=colors[6], label=r'CdTe $\sigma_{pair\,production}$')
#ax.loglog(data[:,0], data[:,5],'m', label='')
ax.loglog(data[:,0], data[:,6],color=colors[7], label=r'CdTe $\sigma_{total}$')
ax.set_xlim(1e-3,1e3)
ax.set_ylim(1e-7, 1e4)
ax.legend(fontsize='small')
#plt.title(r"Attenuation in silicon")
plt.title(r"Attenuation in CdTe")
plt.xlabel(r'Energy $(MeV)$')
plt.ylabel(r'Mass attenuation coefficient $(cm^2 g^{-1})$')
# make it into black and white
#bw.setFigLinesBW(fig)
style.sns.despine()
#plt.show()
plt.savefig('../figures/cdte_cross_sec.pdf')
#plt.savefig('../figures/cdte_cross_sec.png')
