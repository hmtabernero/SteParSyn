#!/usr/bin/python3
# 2-CLAUSE BSD LICENCE
#Copyright 2015-2021 Hugo Tabernero, Emilio Marfil, Jonay Gonzalez Hernandez, and David Montes
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Times']})#,'size': 9})
import matplotlib.pyplot as plt
import _pickle as pic
import numpy as np
from astropy.stats import sigma_clip
import pygtc
#plt.rc('text', usetex=True)

def momenta(xn):
    mu,sigma = np.mean(xn),np.std(xn)
    return mu,sigma

name=input()
fichr=open("BINOUT/"+name+"b.bin","rb")
best = pic.load(fichr)
chain=pic.load(fichr)
fprob=pic.load(fichr)
acc=pic.load(fichr)
fichr.close()

samples=chain.reshape((-1, len(chain[0,0,:])))
#SteParSyn samples Teff/1000. instead of Teff
samples[:,0] = 1000.*samples[:,0]

params = ('$T_\mathrm{eff}$ [K]',
          '$\log{g}$ [dex]',
          '[Fe/H] [dex]',
          '$V_\mathrm{broad}$ [km s$^{-1}$]')
GTC = pygtc.plotGTC(chains=[samples],
                    paramNames=params,
                    nContourLevels =3,
                    smoothingKernel = 0,
                    nBins= 30,
                    colorsOrder=(['blues_old']),
                    customLabelFont={'family':'Times new Roman', 'size':6},
                    plotName='PLOTS/corner_'+name+'.pdf')


Teff,eTeff=momenta(samples[:,0])
logg,elogg = momenta(samples[:,1])
MH,eMH = momenta(samples[:,2])
vbroad,evbroad= momenta(samples[:,3])

print('----------------------------------')
print('Results for '+name)
print('----------------------------------')
print('Teff   = {0:6.0f} +- {1:6.0f} K'.format(Teff,eTeff))
print('log(g) = {0:6.2f} +- {1:6.2f} dex'.format(logg,elogg))
print('[Fe/H] = {0:6.2f} +- {1:6.2f} dex'.format(MH,eMH))
print('Vbroad = {0:6.2f} +- {1:6.2f} km/s'.format(vbroad,evbroad))
print('----------------------------------')
print(' ')
print('The corresponding corner plot is: PLOTS/corner_'+name+'.pdf')
print(' ')
