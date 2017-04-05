# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 15:39:02 2014

@author: davkra
"""

import numpy as np
import os
from scipy.signal import sawtooth, square
import matplotlib.pyplot as plt

import style
import bw

PATH = '/home/davkra/code/geant4/geant4medipix/macros/tubes'
filenames = ['80kVp', '100kVp', '120kVp']
colors = ['c','g','b']
data = np.array([])


fig = plt.figure()
for c,name  in zip(colors, filenames):
    with open(os.path.join(PATH,name)) as f:
        for n, line in enumerate(f):
            if 'point' in line:
                data = np.append(data, np.genfromtxt(line.split()[1:]), axis=0)
    a = data[::2]*1000
    b = data[1::2]
    plt.plot(a,b, c, label=name)
    f.close()
    data = np.array([])
plt.legend()
plt.title(r'X-ray spectrum (XOP)')
plt.xlabel(r'Energy (keV)')
plt.ylabel(r'Relative intensity')
bw.setFigLinesBW(fig)
style.sns.despine()
plt.savefig(os.path.join(style.OUTPATH, "xray_spectrum_bw.pdf"))