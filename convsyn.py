#!/usr/bin/python3
# 2-CLAUSE BSD LICENCE
#Copyright 2015-2021 Hugo Tabernero, Jonay Gonzalez Hernandez, Emilio Marfil, and David Montes
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from astropy.convolution import convolve
from astropy.modeling.functional_models import Voigt1D
from scipy import interpolate, special, mean
import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt

def vlambda(inlamb,vstep):

    vlight=2.99792458e5
    xw2=max(inlamb)-0.1
    xw1=min(inlamb)+0.1
    iw1=np.where(inlamb > xw1)
    iw2=np.where(inlamb > xw2)
    iw1=iw1[0]
    iw2=iw2[0]
    w1=inlamb[iw1[0]+1]
    w2=inlamb[iw2[0]-1]

    npix=np.long((iw2[0]-1)-(iw1[0]+1)*vstep)
    wmid=np.sqrt(w1*w2)
    vave=vlight*(10.**(np.log10(w2/w1)/(npix-1.))-1.)
    dwave=(np.log10(w2)-np.log10(w1))/(npix-1.)

    vwavel=np.log10(wmid)-(dwave*np.arange(np.long(npix/2.)))
    vwaveu=np.log10(wmid)+(dwave*np.arange(np.long(npix/2.)))
    vwavel=10.**vwavel[1:len(vwavel)-1]
    vwavel=np.sort(vwavel)
    vwaveu=10.**vwaveu[1:len(vwaveu)-1]
    vwave=np.concatenate((vwavel,[wmid],vwaveu))

    return vwave

#Warning: The public version of convsyn does not include the CARMENES LSF
def conkern(inlamb, influx, vbroad, ldc, kop, channel=None):
    # You must provide vbroad in km/s.
    # Alternatively, you can also give the resolution.
    if vbroad >= 0.: 
        vstep=1
        vlight=2.99792458e5
        vwave=vlambda(inlamb,vstep)
        xw2=max(inlamb)-0.1
        xw1=min(inlamb)+0.1
        iw1=np.where(inlamb > xw1)
        iw2=np.where(inlamb > xw2)
        iw1=iw1[0]
        iw2=iw2[0]
        w1=inlamb[iw1[0]+1]
        w2=inlamb[iw2[0]-1]
        tck=interpolate.splrep(inlamb,influx,k=3, s=0)
        vflux=interpolate.splev(vwave,tck,der=0)
        npix=np.long((iw2[0]-1)-(iw1[0]+1)*vstep)
        wmid=np.sqrt(w1*w2)
        vave=vlight*(10.**(np.log10(w2/w1)/(npix-1.))-1.)
	
        x1=vwave
        y1=vflux
        if kop == 'g': 
            if vbroad > 1000. and kop == 'g':
                vibr=(vlight/vbroad)/(2.*np.sqrt(2.*np.log(2.)))
            else:
                vibr=vbroad#(2.*np.sqrt(2.*np.log(2.)))
            sigma=vibr*wmid/vlight
            nx1=len(x1)
            dx1=(x1[nx1-1]-x1[0])/float(nx1-1)
            xk = (np.arange(nx1)-nx1/2)*dx1
            a1=0.
            a2=sigma
            zk=(xk-a1)/a2
            a0=1./np.sqrt(2.*np.pi)/sigma
            yk=a0*np.exp(-(zk**2.)/2.)
            #ii = np.where(yk > 0.0)

        elif kop == 'r':
            vrot=vbroad
            xi=vrot/vlight*wmid
            d  = 1.0/(np.pi*(1.0-ldc/3.0))
            c1 = 2.0*(1.0-ldc)*d
            c2 = 0.5*np.pi*ldc*d
            nx1 = len(x1)
            dx1 = (x1[nx1-1]-x1[0])/float(nx1-1)
            xk = (np.arange(nx1)-nx1/2)*dx1
            d2 = 1.0-(xk/xi)**2
            ii = np.where(d2 <= 0.0)
            #xk = xk[ii]
            d2[ii] = 0
            yk = c1*np.sqrt(d2)+c2*d2
 
        elif kop == 'm':
            vmac = vbroad
            xi = vmac/vlight*wmid
            nx1 = len(x1)
            dx1 = (x1[nx1 - 1] - x1[0])/float(nx1 - 1)
            xk = (np.arange(nx1) - nx1/2) * dx1
            xk = (xk/xi)
            spi = np.sqrt(np.pi)
            A = 2./(spi*vmac)
            yk = A*(np.exp(-xk**2)-(spi*abs(xk)*special.erfc(abs(xk))))
        nfact = np.sum(yk)
        if nfact > 0.:
            outflux = convolve(y1, yk/nfact, boundary='fill',fill_value=1.)  
        else:
            outflux = y1
        tck2 = interpolate.splrep(vwave,outflux,k=3, s=0)
        bflux = interpolate.splev(inlamb, tck2, der=0)

    else:
        bflux = influx

    return bflux

