# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 13:05:43 2014

@author: davkra
special fitting library, uses TVRegDiff
"""

from tvregdiff import TVRegDiff
from lmfit.models import GaussianModel, VoigtModel, StepModel, ConstantModel, DonaichModel, LinearModel
#from numpy import NaN, Inf, arange, isscalar, asarray, array
import numpy as np
from scipy.special import erfc
import sys


class LMFit():

    def __init__(self,
                 y,
                 x=None,
                 skipfirst=0,
                 model='',
                 diff=None,
                 plot=False):


        if x is None:
            self.x = np.arange(len(y[skipfirst:]))
        else:
            self.x = np.asarray(x)
        self.y = np.asarray(y)


        if diff is True:
            self.y_diff = -np.diff(self.y[skipfirst:].astype(np.double))
            self.x = self.x[skipfirst+1:]
        else:
            print 'dont diff'
            self.y_diff = self.y[skipfirst:]
            self.x = self.x = self.x[skipfirst:]
            #self.skipfirst = self.skipfirst - 1
        self.model = model
        self.diff = 'standard' # can be tvdiff

    def tvdiff(self,
               alpha,
               it=20,
               dx=0.05,
               ep=1e-6,
               scale = 'large',
               p=0):

        # u = TVRegDiff( data, iter, alph, u0, scale, ep, dx, plotflag, diagflag );
        self.y_diff = TVRegDiff(self.y, it, alpha, dx=dx, ep=ep, scale=scale, plotflag=p)

    def peakdet(self, delta, x=None):
        """
        Converted from MATLAB script at http://billauer.co.il/peakdet.html

        Returns two arrays

        function [maxtab, mintab]=peakdet(v, delta, x)
        %PEAKDET Detect peaks in a vector
        %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        %        maxima and minima ("peaks") in the vector V.
        %        MAXTAB and MINTAB consists of two columns. Column 1
        %        contains indices in V, and column 2 the found values.
        %
        %
        %        A point is considered a maximum peak if it has the maximal
        %        value, and was preceded (to the left) by a value lower by
        %        DELTA.

        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.

        """
        maxtab = []
        mintab = []

#        if x is None:
#            x = np.arange(len(self.y_diff))
#        else:
#            x = self.x

        if len(self.y_diff) != len(self.x):
            sys.exit('Input vectors v and x must have same length')

        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')

        if delta <= 0:
            sys.exit('Input argument delta must be positive')

        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN

        lookformax = True

        for i in xrange(len(self.y_diff)):
            this = self.y_diff[i]
            if this > mx:
                mx = this
                mxpos = self.x[i]
            if this < mn:
                mn = this
                mnpos = self.x[i]

            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = self.x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = self.x[i]
                    lookformax = True

        return np.array(maxtab), np.array(mintab)

    def fit(self,model='gauss'):

        maxtab, mintab = self.peakdet(self.y_diff.max()/2,self.x)
        print maxtab
        if model == 'gauss':
            self.mod = GaussianModel()
            self.mod.guess_starting_values(self.y_diff,x=self.x)
            out = self.mod.fit(self.y_diff, x=self.x)
            self.out = out

        elif model == 'gauss2':
            gauss1 = GaussianModel(prefix='g1_')
            gauss2 = GaussianModel(prefix='g2_')


            #gauss1.set_param('center',  maxtab[0][0]+self.skipfirst)#  , min=maxtab[0][0]-10, max=maxtab[0][0]+10)
            gauss1.set_param('center', 85)
            gauss1.set_param('sigma',      7, min=1)
            gauss1.set_param('amplitude', maxtab[0][1], min=10)
            #print maxtab[0][0]+self.skipfirst+15
            gauss2.set_param('center',    maxtab[0][0]-15)#, min=maxtab[0][0], max=maxtab[0][0]+5)
            gauss2.set_param('sigma',      7, min=1)
            gauss2.set_param('amplitude', maxtab[0][1]/7, min=250)
            self.mod = gauss1 + gauss2
            out = self.mod.fit(self.y_diff, x=self.x)
            self.out = out

        elif model == 'voigt':
            mod = VoigtModel()
            mod.set_param('center', maxtab[0][0])
            mod.set_param('sigma', 7)
            self.mod =  mod
            # self.mod.guess_starting_values(self.y_diff, x=self.x[self.skipfirst+1:])
            print self.y_diff.shape, self.x
            out = self.mod.fit(self.y_diff, x=self.x)
            self.out = out

        elif model == 'lvoigt':
            lin = LinearModel(prefix='linear_')
            lin.guess_starting_values(self.y_diff, x=self.x)
            #mod = VoigtModel(prefix='voigt_')
            mod = VoigtModel()
            mod.guess_starting_values(self.y_diff, x=self.x)
            #mod.set_param('center', maxtab[0][0])
            #mod.set_param('sigma', 7)
            self.mod = mod + lin
            out = self.mod.fit(self.y_diff, x=self.x)
            self.out = out

        elif model == 'svoigt':
            step = StepModel(prefix='step_', form='erf')
            step.set_param('center', maxtab[0][0]-10)
            step.set_param('amplitude', 5000, max=6000)
            #mod = VoigtModel(prefix='voigt_')
            mod = VoigtModel(prefix='gauss')
            mod.set_param('center', maxtab[0][0])
            mod.set_param('sigma', 7)
            self.mod = mod + step
            # self.mod.guess_starting_values(self.y_diff, x=self.x[self.skipfirst+1:])
            out = self.mod.fit(self.y_diff, x=self.x)
            self.out = out

        elif model == "erf":
            step = StepModel(form='erf')
            step.guess_starting_values(self.y, self.x)
            offset = ConstantModel()
            offset.set_param('c', self.y.min())
            self.mod = step + offset

            out = self.mod.fit(self.y, x=self.x)
            self.out = out
        elif model == 'donaich':
            #step = StepModel(form='erfc')
            #step.set_param("center", 7)
            don = DonaichModel()
            don.guess_starting_values(self.y_diff, x=self.x)
            mod = don #+ step
            self.mod = mod
            out = mod.fit(self.y_diff, x=self.x)
            self.out = out
        else:
            print "Choose a function!"
            pass


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpxplot import Frame
    import os

    #PATH='/home/davkra/analysis/Geant4Medipix/measurements/E05_W0015/ToT'
    PATH="/home/davkra/analysis/Geant4Medipix/measurements/W115_D10/Cu_CSM"
#    PATH='/home/davkra/analysis/Geant4Medipix/measurements/W115_H9/Ag_repeat/'
    #metal = 'Ag'
    metal = 'Cu'
    skip = 1
#
#    # load frames
    A = Frame.Frame()
    A.loadDacScan(os.path.join(PATH))
    #ToT--------------
#    A.load(os.path.join(PATH,metal))
#    y = A.data['C']
#    y,bin_edges = np.histogram(y[y<250], bins=125)
#    x = bin_edges[0:-1]
    # ---------------

#    # extract dac scan
    x = A.th[1:]
    y = A.getCountsPerFrame()[1:]


#    # special syntax for python measurements
#    PATH='/home/davkra/analysis/Geant4Medipix/measurements/W115_H9/python/'
#    data = np.load(os.path.join(PATH,"cu_CSM.dat.npy"))
#    th = np.loadtxt(os.path.join(PATH, "cu_CSM.dat"))
#    y = data.sum(axis=2).sum(axis=1)
#    x = th

    fit = LMFit(y, x=x, skipfirst=skip, diff=True)#, x=x, skipfirst=skip)
#    fit.tvdiff(0.025e-2,it=200, dx=0.05, ep=1e-3, scale='large', p=0)

    maxtab, mintab = fit.peakdet(fit.y_diff.max()/2)

    #fit.fit('voigt')
    fit.fit('gauss')

    plt.plot(x[2:],fit.y_diff, 'o')
    #plt.scatter(x[skip+1:][np.array(maxtab)[:,0].astype(np.int)], np.array(maxtab)[:,1], color='blue')
    #plt.scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red')
    plt.plot(x[2:],fit.out.best_fit)
    #comp_gauss = fit.mod.components[0].eval(x=x)
    #comp_line  = fit.mod.components[1].eval(x=x)
    #plt.plot(x,comp_gauss, 'r--')
    #plt.plot(x,comp_line, 'g--')
    plt.show()
    print fit.mod.fit_report()

    #print metal, fit.mod.result.values['center'],  fit.mod.result.values['sigma'], fit.mod.result.values['fwhm']
    print metal, fit.mod.result.values['center'],  fit.mod.result.values['sigma'], fit.mod.result.values['sigma']*2.35482, fit.mod.result.values['sigma']*2.35482/fit.mod.result.values['center']*100
    #print metal, fit.mod.result.values['g2_center'],  fit.mod.result.values['g2_sigma'], fit.mod.result.values['g2_fwhm']