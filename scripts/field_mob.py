# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:46:47 2014

@author: davkra
"""

import bw
import style
import numpy as np
import matplotlib.pyplot as plt

T = 300
vs_e = 1.53e9*T**-0.87
Ec_e = 1.01*T**1.55
beta_e = 2.67e-2*T**0.66

vs_h = 1.62e8*T**-0.52
Ec_h = 1.24*T**1.68
beta_h = 0.46*(T**0.17)

E = np.logspace(0.1, 5., 100, dtype=np.float128)

mu_e = vs_e / Ec_e / (1. + (E / Ec_e)**beta_h)**(1.0 / beta_h)
mu_h = vs_h / Ec_h / (1. + (E / Ec_h)**beta_h)**(1.0 / beta_h)
#
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = plt.subplot(111)

ax.loglog(E, mu_e, 'k', label='$\mu_e$')
ax.loglog(E, mu_h, 'k', label='$\mu_h$')
#
plt.legend()

l = ax.get_lines()
l[1].set_dashes([10,3])
l = ax.get_legend().get_lines()
l[1].set_dashes([10,3])
ax.grid(b=True, which='minor', color='w', linewidth=0.5)
#
ax.set_title("Field dependent mobility (Si)")
ax.set_xlabel(r"Electric field \si{(\volt\per\cm)}")
ax.set_ylabel(r"Mobility \si{(\cm^2\per\volt\per\second)}")
style.sns.despine()
#plt.show()
plt.savefig("../figures/field_mob.pdf")


