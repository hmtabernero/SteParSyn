#!/usr/bin/python3
# 2-CLAUSE BSD LICENCE
#Copyright 2015-2021 Hugo Tabernero, Emilio Marfil, Jonay Gonzalez Hernandez, and David Montes
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from numpy.polynomial.chebyshev import chebfit,chebval    
from scipy.stats import sigmaclip
import pickle as pic
import numpy as np
import emcee,george
from celerite import terms,GP
import astropy.io.ascii as at
from scipy.interpolate import griddata, splrep, splev
from scipy.optimize import minimize
from multiprocessing import Pool
import convsyn as cs
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import extinction
import os
vlight       = 299792.458
import time
from os import system
from scipy.optimize import least_squares,curve_fit
import matplotlib.pyplot as plt
os.environ["OMP_NUM_THREADS"] = "1"


def brew16_cal(Teff,logg,MH):
    Teff = 1000.*Teff
    if logg >= 4.0:
        vmac = 1.3+2.202*np.exp(0.0019*(Teff-5777))
    elif (logg < 4.0) and logg >= 3.0:
        vmac = 3.3+1.166*np.exp(0.0028*(Teff-5777))
    else:
        vmac = 4.0
    return vmac

def doyle14_cal(Teff,logg,MH):
    Teff = Teff*1000.
    vmac = 3.21 + 2.33*10**(-3.)*(Teff-5777.)+ 2.*10**(-6.)*(Teff-5777.)**2.-2.*(logg-4.44)
    return vmac

def apo_giant(Teff,logg,MH):
    logvmac=0.470794-0.254120*MH
    return 10.**logvmac

def A_lambda(A_v,wave): 
    log_wave = np.log10(wave)-4. 
    log_A_l = 0.61-2.22*log_wave + 1.21*log_wave**2. 
    A_fact_1 = extinction.fitzpatrick99(np.array([12500.]), A_v,3.1, unit='aa') 
    A_fact_2 = 10.**(0.61-2.22*np.log10(1.25)+1.21*np.log10(1.25)**2.) 
    A_l_nir = A_fact_1*10.**(log_A_l)/A_fact_2 
    A_l_tot = extinction.fitzpatrick99(wave, A_v,3.1, unit='aa') 
    A_l_tot[wave >= 12500.] = A_l_nir[wave >= 12500.] 
    return A_l_tot 

class CosTerm(terms.Term):
    parameter_names = ("log_P",)
    def get_real_coefficients(self, params):
        log_P, = params
        return (0.5,1.,)

    def get_complex_coefficients(self, params):
        log_P, = params
        return (0.5,0.0,2*np.pi*np.exp(-log_P),)

def read_opt(name):
    opt_file = at.read('OPT/'+name+'.opt')
    opt_pars = opt_file['VALUES']
    flux_log_scale = opt_pars[0]
    error_map      = opt_pars[1]
    resol_wave     = float(opt_pars[2])
    threshold      = float(opt_pars[3])
    nwalk_fact     = np.compat.long(opt_pars[4])
    nburn          = np.compat.long(opt_pars[5])
    nsteps         = np.compat.long(opt_pars[6])
    range_file     = opt_pars[7]
    mask_file      = opt_pars[8]
    config_file    = opt_pars[9]
    name_grid      = opt_pars[10]
    gp_type        = opt_pars[11]
    gp_active      = opt_pars[12]
    prior_type     = opt_pars[13]
    return flux_log_scale,error_map,nwalk_fact,nburn,nsteps,range_file,mask_file,config_file,name_grid,gp_type,gp_active,prior_type,resol_wave,threshold

def read_config(name):
    param_file = at.read('CONFIG/'+name+'.conf')
    vsini_type = param_file['TYPES'][param_file['NAMES'] == 'vsini']
    vmac_type  = param_file['TYPES'][param_file['NAMES'] == 'vmac']
    resol_type = param_file['TYPES'][param_file['NAMES'] == 'Resolution']
    types = np.array([vmac_type,vsini_type,resol_type])
    fixed_vec     = np.array(param_file['FIXED'])
    fixed_vec_num = np.zeros(len(fixed_vec))
    for i in range(len(fixed_vec)):
        if fixed_vec[i] == 'True':
            fixed_vec_num[i] = 1
    upper_bounds = np.array(param_file['UPPER_BOUND'])
    lower_bounds = np.array(param_file['LOWER_BOUND'])
    mu           = np.array(param_file['MU'])
    sigma        = np.array(param_file['SIGMA'])
    par_dict = { 'parameters': np.array(param_file['NAMES']), 'fixed': fixed_vec_num, 'values': np.array(param_file['VALUES']), 'convol_types': types, 'upper_bounds': upper_bounds, 'lower_bounds': lower_bounds, 'mu':mu, 'sigma':sigma} 
    return par_dict

def turn_params(params,dict_params):                                                     
    full_params  = dict_params['values']
    fixed_params = dict_params['fixed']
    full_params[fixed_params < 1] = params
    Teff,logg,MH,RV,resolution,vsini,vmac,ldc,omega,A_v,log_sigma,log_rho = full_params
    vmac_type,vsini_type,resol_type = dict_params['convol_types']
    return Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type


def get_params(dict_params):
    full_params  = dict_params['values']                                                                                                                                                                    
    fixed_params = dict_params['fixed']
    upper_params = dict_params['upper_bounds']
    lower_params = dict_params['lower_bounds']
    mu_params    = dict_params['mu']
    sigma_params = dict_params['sigma']
    sel_par = fixed_params < 1
    return full_params[sel_par],upper_params[sel_par],lower_params[sel_par],mu_params[sel_par],sigma_params[sel_par]

def rebin_spec(input_spec,wave_grid,resolution,vsini,ldc,vmac,vmac_type,vsini_type,resol_type):
     interpolated_flux = np.array([])
     for i in range(len(central_wave_rank)):
         sel_syn = tuple([np.abs(wave_grid       - central_wave_rank[i]) <= width_wave_rank[i]])  # Range in synthetic spectra
         sel_obs = tuple([np.abs(wave_obs_rank  - central_wave_rank[i]) <= width_wave_rank[i]])  # Range in real spectra
         if len(wave_obs_rank[sel_obs]) > 0:
            rank_wave_grid = wave_grid[sel_syn]
            rank_flux_syn  = input_spec[sel_syn]
            rank_wave_obs  = wave_obs_rank[sel_obs]
            rank_flux_obs  = flux_obs_rank[sel_obs]
            rank_eflux_obs = eflux_obs_rank[sel_obs]
            bsyn_flux = broad_spec(rank_flux_syn,rank_wave_grid,resolution,vsini,ldc,vmac,vmac_type,vsini_type,resol_type)
            tck2 = splrep(rank_wave_grid, bsyn_flux, k=3, s=0)  
            rbsyn_flux = splev(rank_wave_obs, tck2, der=0)
            if rank_order[i] >= 0:
                nbrsyn_flux = normalize_spec(rbsyn_flux,rank_flux_obs,rank_eflux_obs,rank_wave_obs,rank_order[i])
            else:
                nbrsyn_flux = rbsyn_flux
            interpolated_flux = np.append(interpolated_flux, nbrsyn_flux)  # Interpolate synthetic wave to obseved wave
     return interpolated_flux

def normalize_spec(synthetic_flux,observed_flux,observed_eflux,observed_wave,order):
    Mwave = max(observed_wave)
    mwave = min(observed_wave)
    mean_wave = 0.5*(Mwave+mwave)
    half_wave = 0.5*(Mwave-mwave)
    if flux_log_scale == 'True':
        sel = (np.abs(observed_wave-mean_wave) < half_wave-0.5)
    else:
        sel = (np.abs(observed_wave-mean_wave) < half_wave-0.5) & (synthetic_flux > 0.) & (observed_flux > 0.)
    
    if order > 0:
        
        if flux_log_scale == 'True':                                                                                                                                                                     
            residual = observed_flux[sel]-synthetic_flux[sel]                                                                                                                                                 
        else:
            residual = np.log(observed_flux[sel]/synthetic_flux[sel]) 
        
        continuum = chebval(observed_wave,chebfit(observed_wave[sel],residual,order))
    else:
        if flux_log_scale == 'True':
            continuum =  np.median(observed_flux[sel])-np.median(synthetic_flux[sel])
        else:
            selsyn = synthetic_flux[sel] >= np.percentile(synthetic_flux[sel],75)
            selobs = observed_flux[sel]  >= np.percentile(observed_flux[sel],75)
            osel,ssel = observed_flux[sel], synthetic_flux[sel]
            continuum = np.median(np.log(osel[selobs]))-np.median(np.log(ssel[selsyn]))
    
    if flux_log_scale == 'True':
        return synthetic_flux+continuum
    else:
        return synthetic_flux*np.exp(continuum)

    
def broad_spec(model_spec,wave_spec,resolution,vsini,ldc,vmac,vmac_type,vsini_type,resol_type):
      if resol_type =='v':
          if wave_spec[0] >= 9600.:
             channel = 'nir'
          else:
             channel = 'vis'
      else:
          channel = None
      if vmac > 0:
          if vmac_type == 'brew16' or vmac_type == 'apo_giant' or vmac_type == 'doyle14':
             vmac_type = 'm'
          mmodel_spec  =  cs.conkern(wave_spec,model_spec,vmac,0.0,vmac_type)
      else:
          mmodel_spec = model_spec
      if vsini> 0.:
          rmmodel_spec  = cs.conkern(wave_spec,mmodel_spec,vsini,ldc,vsini_type)
      else:
          rmmodel_spec  = mmodel_spec
      if  resolution > 0.:
          brmmodel_spec = cs.conkern(wave_spec,rmmodel_spec,resolution,0.0,resol_type,channel=channel)
      else:
          brmmodel_spec = rmmodel_spec
      return brmmodel_spec

def syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type):
    syn_spectrum = np.array([])
    test_int = False
    sel = np.where((np.abs(Teff_grid - Teff) <= 0.3))
    if len(np.unique(Teff_grid[sel])) < 2:
        sel = np.where((np.abs(Teff_grid - Teff) <= 0.5))
    if vmac_type == 'brew16':
        vmac = brew16_cal(Teff,logg,MH)
    elif vmac_type == 'doyle14':
        vmac = doyle14_cal(Teff,logg,MH)
    elif vmac_type == 'apo_giant':
        vmac = apo_giant(Teff,logg,MH)
    if np.shape(sel)[1] > 1  and np.abs(RV) <= 1000. and vsini >=  0. and vmac >= 0. and resolution >= 0. and ldc >= 0 and A_v >= 0.:
        testt = (min(Teff_grid[sel]) <= Teff) & (max(Teff_grid[sel]) >= Teff) & (max(Teff_grid[sel]) != min(Teff_grid[sel]))
        testg = (min(logg_grid[sel]) <= logg) & (max(logg_grid[sel]) >= logg) & (max(logg_grid[sel]) != min(logg_grid[sel]))
        testM = (min(MH_grid[sel]) <= MH)  & (max(MH_grid[sel]) >= MH) & (max(MH_grid[sel]) != min(MH_grid[sel]))
        if (testt == True) and (testg == True) and (testM == True):
            intwei = griddata((Teff_grid,logg_grid,MH_grid), gridw, (Teff, logg, MH), method='linear', fill_value=np.nan, rescale=True)
            if np.isfinite(intwei[0]) == True:
                model_spec = mu_grid + (sig_grid * np.dot(intwei, egridw))
                test_int = True
                wave_grid_rv = wave_grid*(1.+RV/299792.458)
                if A_v > 0.:
                    A_l = A_lambda(A_v,wave_grid_rv)
                else:
                    A_l = 0.
                if flux_log_scale == 'True':
                    model_spec = model_spec - 0.4*A_l+omega
                else:
                    model_spec = model_spec*10.**(-0.4*A_l+omega)
                nrbsyn_flux = rebin_spec(model_spec, wave_grid_rv,resolution,vsini,ldc,vmac,vmac_type,vsini_type,resol_type)
    if test_int == True:
        return test_int,nrbsyn_flux
    else:
        return test_int,test_int

def masked_syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type):
    test_int,nbsyn_flux = syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type)
    syn_spectrum =np.array([])
    for i in range(len(central_wave_mask)):
        sel_mask = tuple([np.abs(wave_obs_rank - central_wave_mask[i]) <= width_wave_mask[i]])                                                      
        obs_flux_mask = flux_obs_rank[sel_mask]
        if test_int == True:
            syn_flux_mask = nbsyn_flux[sel_mask]
            syn_spectrum = np.append(syn_spectrum,syn_flux_mask)
        else:
            syn_spectrum = np.append(syn_spectrum,np.zeros(len(obs_flux_mask))+np.nan)
    return syn_spectrum


def residuals(params,dict_params,flux,eflux):
    Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type =  turn_params(params,dict_params)
    model = masked_syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type)
    if np.isfinite(model[0]) == True:
        return (model-flux)/eflux
    else:
        return np.zeros(len(model))-1000000.

def get_syn(wave,*params):
    Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type =  turn_params(params,dict_params)                                                            
    model = masked_syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type)                                                                                                          
    if np.isfinite(model[0]) == True:
        return model
    else:
        model = np.ones(len(model))
        model[0] = 10.*100.
        return model

def do_error_map(centers,sigmas,amplitudes,sig_res):
    extra_err      = np.zeros(len(masked_wave_obs))
    extra_err_rank = np.zeros(len(wave_obs_rank))
    for  i in range(len(central_wave_rank)):
        sel_rank = tuple([np.abs(wave_obs_rank - central_wave_rank[i]) <= width_wave_rank[i]])
        for j in range(len(amplitudes)):
            extra_err_rank[sel_rank] +=  amplitudes[j]*np.exp(-0.5*((centers[j]-wave_obs_rank[sel_rank])/sigmas[j])**2.)
        extra_err_rank[sel_rank] = np.sqrt(extra_err_rank[sel_rank]**2.+sig_res[i]**2.)
    extra_err_rank_added = np.array([max([extra_err_rank[i],eflux_obs_rank[i]])  for i in range(len(eflux_obs_rank))])
    dummy,dummy,extra_err,dummy,dummy =  mask_flux(wave_obs_rank,flux_obs_rank,extra_err_rank)
    
    return  extra_err_rank,extra_err

def readrange(name):
    ranges = at.read('RANGES/'+name+'_ranges.txt')
    lowerR = ranges['l1']
    upperR = ranges['l2']
    order  = ranges['order']
    typeR  = ranges['type']
    meanR = 0.5 * (lowerR + upperR)
    widthR = 0.5 * (upperR - lowerR)
                                         
    return meanR, widthR,typeR,order

def readmask(name):
    masks = at.read('MASKS/'+name+'_masks.txt', delimiter=' ')
    meanM = masks['center']
    widthM = masks['width']
    return meanM, widthM


def read_spectrum(fspectrum, meanR, widthR):
    spectrum = at.read('SPECTRA/' + fspectrum + '.txt')
    try:
        swave = spectrum['wavelength']
    except:
        swave = spectrum['waveobs']
    sflux = spectrum['flux']
    sfluxerr = spectrum['err']
    trimswave = np.array([])
    trimsflux = np.array([])
    trimsfluxerr = np.array([])

    for indmeanR, indwidthR in zip(meanR, widthR):
        sel_rank = np.abs(swave - indmeanR) <= indwidthR
        wave_rank  = swave[sel_rank]
        sflux_rank = sflux[sel_rank]
        esflux_rank = sfluxerr[sel_rank]
        trimswave    = np.append(trimswave, wave_rank)
        trimsflux    = np.append(trimsflux, sflux_rank)
        trimsfluxerr = np.append(trimsfluxerr, esflux_rank)
    return trimswave, trimsflux,trimsfluxerr


def write_spec(ldo,flux,eflux,fspectrum):
    at.write([ldo,flux,eflux],"eSPECTRA/"+ fspectrum+'.txt',names=['wavelength','flux','err'],overwrite=True)
    return 0

def read_grid(fgrid):
    g = open('GRIDS/' + fgrid +'.bin', 'rb')
    teffg = 0.001*np.array(pic.load(g))
    loggg = np.array(pic.load(g))
    metalg = np.array(pic.load(g))
    ldog = np.array(pic.load(g))
    gridwg = np.transpose(np.array(pic.load(g)))
    egridwg = np.array(pic.load(g))
    mugridg = np.array(pic.load(g))
    siggridg = np.array(pic.load(g))
    g.close()
    return teffg, loggg,metalg, ldog, gridwg, egridwg, mugridg, siggridg



def print_result(params,dict_params):
    Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type  =  turn_params(params,dict_params)
    string1 =  ('Teff: {0:5.3f}, log(g): {1:4.2f}, [M/H]: {2:5.2f}, RV: {3:5.1f}\n').format(Teff,logg,MH,RV)
    string2 =  ('log_sigma: {0:5.2f}, log_rho: {1:5.2f}\n').format(log_sigma,log_rho)
    string3 =  ('Resolution: {0:12.2f}, vsini: {1:5.2f}, ldc: {2:5.2f}, vmac:{3:5.2f}\n').format(resolution,vsini,ldc,vmac)
    string4 =  ('Omega: {0:6.3f}, Av: {1:5.2f}\n').format(omega,A_v)
    return string1+string2+string3+string4

def mask_flux(wave,flux,eflux):
    flux_masked  = np.array([])
    eflux_masked = np.array([])
    wave_masked  = np.array([])
    velo_masked  = np.array([])
    for i in range(len(central_wave_mask)):                                                                                                                                                             
        sel_mask = tuple([np.abs(wave_obs_rank - central_wave_mask[i]) <= width_wave_mask[i]])                                                                                                          
        flux_masked  = np.append(flux_masked,flux[sel_mask])
        eflux_masked = np.append(eflux_masked,eflux[sel_mask])
        wave_masked  = np.append(wave_masked,wave[sel_mask])
    wmid  = np.sqrt(np.max(wave_masked)*np.min(wave_masked))
    dwmid = (np.max(wave_masked)-np.min(wave_masked))/(len(wave_masked)-1)
    velo_masked  = vlight*(wave_masked-wmid)/wmid
    velo_ranked  = vlight*(wave-wmid)/wmid
    return wave_masked,flux_masked,eflux_masked,velo_masked,velo_ranked

def log_prior(Teff,logg,MH,RV,vsini,vmac):
    if prior_type == 'dwarfs':
        tck2 = splrep(Teff_dwarfs, logg_dwarfs, k=3, s=0)
        logg_int = splev(1000.*Teff, tck2, der=0)
        sigma_logg = 0.2
        lprior = -0.5*((logg_int-logg)/sigma_logg)**2.
    else:
        lprior = 0.
    return lprior

def check_bounds(value,upper,lower,mu,sigma):
    for  i in range(len(value)):
        if value[i] > upper[i] or value[i] < lower[i]:
            return -np.inf
    log_prior = 0.
    for i in range(len(value)):
        if sigma[i] > 0.:
            log_prior += -0.5*((value[i]-mu[i])/sigma[i])**2.-np.log(sigma[i])

    return log_prior

    


def log_probability(params,dict_params):
    Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type =  turn_params(params,dict_params)
    log_lik = -np.inf
    lprior_bounds = check_bounds(params,upper_bounds,lower_bounds,mu_prior,sigma_prior)
    lprior_params = log_prior(Teff,logg,MH,RV,vsini,vmac)
    if np.isfinite(lprior_params) == True and np.isfinite(lprior_bounds) == True:
        masked_flux_syn = masked_syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type)
        if np.isfinite(masked_flux_syn[0]) and np.abs(log_rho) <= 1000. and  np.abs(log_sigma) <= 1000.:
            res     = (masked_flux_obs-masked_flux_syn)
            log_lik = lprior_params+lprior_bounds
            if gp_active == 'yes':
                for i in range(len(central_wave_mask)):                                                                                                                                                     
                    sel_mask = tuple([np.abs(masked_wave_obs - central_wave_mask[i]) <= width_wave_mask[i]])
                    mean_dist = width_wave_mask[i]/(len(masked_flux_obs[sel_mask])-1)
                    if mean_dist > 0:
                        log_len    = log_rho+np.log(mean_dist)
                        median_val = np.log(np.median(masked_flux_obs)) 
                        log_s      = log_sigma+median_val
                        if gp_type == 'matern32':
                            kernel = terms.Matern32Term(log_sigma=log_s,log_rho=log_len)
                        elif gp_type == 'matern32cos':
                            kernel = terms.Matern32Term(log_sigma=log_s,log_rho=log_len)*CosTerm(log_P=np.log(8.)+log_len)                                                                          
                        elif gp_type == 'exp':
                            kernel = terms.RealTerm(log_a=2.*log_s,log_c=-log_len)
                        elif gp_type == 'jitter':                                                                                                                                                      
                            kernel = terms.JitterTerm(log_sigma=log_s)
                        gp = GP(kernel)
                        vec = masked_wave_obs[sel_mask]
                        gp.compute(vec,err_tot[sel_mask])
                        log_lik = log_lik+gp.log_likelihood(res[sel_mask],quiet=True)
            else:
                if gp_type == 'prop' or gp_type =='none':
                    var  = (np.exp(log_sigma)*masked_flux_obs)**2.+err_tot**2.
                elif gp_type == 'errfact':
                    var = np.exp(2.*log_sigma)*err_tot**2.
                elif gp_type == "chifact":
                    var = np.exp(2.*log_sigma)*np.abs(masked_flux_obs)
                else:
                    var = err_tot**2.
                res2 = res*res/var
                log_lik = -0.5*(np.sum(res2+np.log(var)))+lprior_params+lprior_bounds
    return log_lik

def neg_log_like(params,dict_params):
    log_lik = log_probability(params,dict_params)
    if np.isfinite(log_lik) == True:
        return -log_lik
    else:
        return 10.**300.

def fit_gauss(x,y,yerr):
    model_gauss = models.Gaussian1D()
    fitter_gauss = fitting.LevMarLSQFitter()
    best_fit_gauss = fitter_gauss(model_gauss, x, y, weights=1./yerr)
    amp,mu,sigma = best_fit_gauss.amplitude.value,best_fit_gauss.mean.value,best_fit_gauss.stddev.value
    return amp,mu,sigma


def get_error(syn_flux,flux,error_flux,wave,resol_wave,threshold):
    residuals =  syn_flux-flux
    mu_res = np.median(residuals)
    sig_res = 1.4826*np.median(np.abs(residuals-mu_res))
    mask = np.abs(residuals-mu_res) <= sig_res*threshold 
    tentative_peaks = wave[mask]
    covered = np.zeros_like(tentative_peaks, dtype=bool)
    abs_resid = np.abs(residuals)
    selected_mu = np.array([])
    centers = np.array([])
    sigmas  = np.array([])
    amplitudes = np.array([])
    mean_dist = (np.max(wave)-np.mean(wave))/(len(wave)-1)
    
    for wavel, resid in sorted(zip(tentative_peaks, abs_resid), key=lambda t: t[1], reverse=True):
        if wavel in tentative_peaks[covered]:
            continue
        selected_mu = np.append(selected_mu,wavel)
        ind = (tentative_peaks >= (wavel - resol_wave)) & (tentative_peaks <= (wavel + resol_wave))
        covered |= ind
    for mu in selected_mu:
        peak_mask = (wave > mu - resol_wave) & (wave < mu + resol_wave)
        mres  = np.abs(residuals[peak_mask]-mu_res)
        med_res = residuals[peak_mask]
        waves = wave[peak_mask]
        m  = np.sum(mres*waves)/np.sum(mres)
        s  = np.sqrt(mean_dist**2+(np.sum(mres*(waves-m)**2.)/np.sum(mres)))
        a  = np.max(np.abs(med_res))
        if np.abs(a) >= threshold*sig_res:
            centers = np.append(centers,m)
            sigmas  = np.append(sigmas,s)
            amplitudes = np.append(amplitudes,a)
    return centers,sigmas,amplitudes,sig_res

def wide_errors(params,dict_params,obs_flux,error_flux,resol_wave,threshold):
    Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type  =  turn_params(params,dict_params)
    viable, syn_spectrum = syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type)
    if viable == False:
        return flux,error_flux
    else:
        centers = np.array([])
        sigmas  = np.array([])
        sig_res = np.array([])
        amplitudes = np.array([])
        for i in range(len(central_wave_rank)):
            sel_flux = tuple([np.abs(wave_obs_rank- central_wave_rank[i]) <= width_wave_rank[i]])  # Range in synthetic spectra
            m,s,a,sr = get_error(syn_spectrum[sel_flux],obs_flux[sel_flux],error_flux[sel_flux],wave_obs_rank[sel_flux],resol_wave,threshold)
            centers = np.append(centers,m)
            sigmas  = np.append(sigmas,s)
            amplitudes = np.append(amplitudes,a)
            sig_res    = np.append(sig_res,sr)
        return centers,sigmas,amplitudes,sig_res


def optimize_func(initial_point,dict_params,method="Powell",options={'xtol':1e-2, 'ftol': 1e-2,'disp':True}):
    soln = minimize(neg_log_like, initial_point, method=method,options=options, args=(dict_params))
    string1 = print_result(soln.x,dict_params)
    string2 =("log-likelihood: {0}\n".format(-soln.fun))
    return soln.x,soln.fun,string1+string2

def do_curve_fit(initial_point,flux,eflux,wave,tol):
    param_result  , cov  = curve_fit(get_syn, wave, flux, p0=initial_point, sigma=eflux, absolute_sigma=False,method='trf',bounds=(lower_bounds,upper_bounds),xtol=tol,tr_solver='lsmr',loss='huber')
    param_resultp , covp = curve_fit(get_syn, wave, flux, p0=initial_point, sigma=eflux, absolute_sigma=True,method='trf',bounds=(lower_bounds,upper_bounds),xtol=tol,tr_solver='lsmr',loss='huber')
    err1 = np.sqrt(np.diag(cov))
    err2 = np.sqrt(np.diag(covp))
    err_fact = np.mean(err1/err2)
    return param_result,cov,err_fact

def set_walkers(initial,svec,nwalkers,ndim):
    nfact = np.compat.long(nwalkers/ndim)
    p0 = []
    for i in range(ndim):
        dvec = np.zeros(ndim)
        dvec[i] = svec[i]
        for j in range(nfact):
            p0.append(initial+2.*svec*(np.random.random_sample(ndim)-0.5))# for i in range(nwalkers)]
    print(svec)
    f0 = [log_probability(p0[i], dict_params) for i in range(nwalkers)]
    sel = np.where(np.isfinite(f0))[0]
    pgood = [p0[sel[i]] for i in range(len(sel))]  # p values for which f is finite.
    for i in range(nwalkers):
        if not np.isfinite(f0[i]):
            p0[i] = pgood[np.random.randint(len(pgood) - 1)]
    return p0

def domcmc(initial,nwalkers,nburn,nsteps,ndim,svec,pool):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,args=[dict_params],pool=pool)#,moves=emcee.moves.DESnookerMove())
    p0 = set_walkers(initial,svec,nwalkers,ndim)
    if nburn > 0:
        print("Running burn-in...")
        p0, lp, _ = sampler.run_mcmc(p0, nburn,progress=True)
        burnp = p0[np.argmax(lp)]
        print("Best After BURN..")
        print(print_result(burnp,dict_params))
        sampler.reset()
    else:
        burnp = initial
    print("Running production...")
    sampler.run_mcmc(p0,nsteps,progress=True)
    return sampler.chain, sampler.lnprobability, sampler.acceptance_fraction

prior_dwarfs  =  at.read("prior_dwarfs")
Teff_dwarfs = np.flip(prior_dwarfs['Teff'])
logg_dwarfs = np.flip(prior_dwarfs['logg'])
name_star = input()                                                                                                                                 
opt_file  = input()                                                                                                                                              
method    = input()
flux_log_scale,error_map,nwalk_fact,nburn,nsteps,range_file,mask_file,config_file,name_grid,gp_type,gp_active,prior_type,resol_wave,threshold =  read_opt(opt_file) 

if range_file == 'STAR':
    range_file = name_star
if mask_file == 'STAR':
    mask_file = name_star
if config_file == 'STAR':
    config_file = name_star

print('Star under analysis: '+name_star)                                                                                                                                                                    
print('Setting initial parameters')                                                                                                                                                                         
dict_params   = read_config(config_file)                                                                                                                                                                      
print('Reading Synthetic PCA grid')
Teff_grid, logg_grid, MH_grid, wave_grid, gridw, egridw, mu_grid, sig_grid = read_grid(name_grid)

print('Reading line ranges')
central_wave_rank, width_wave_rank, rank_type, rank_order = readrange(range_file)
print('Reading line masks')
central_wave_mask, width_wave_mask = readmask(mask_file)
print('Reading observed spectrum')
wave_obs_rank, flux_obs_rank, eflux_obs_rank = read_spectrum(name_star,central_wave_rank,width_wave_rank)
print('Observed spectrum was cut into spectral ranges')
masked_wave_obs,masked_flux_obs,masked_eflux_obs,masked_velo_obs,ranked_velo_obs =  mask_flux(wave_obs_rank,flux_obs_rank,eflux_obs_rank)
print('Observed spectrum was cut into line masks')
extra_err      = np.zeros(len(masked_wave_obs))
extra_err_rank = np.zeros(len(wave_obs_rank))



if gp_type =='jitter' or gp_active == 'no':
    dict_params['fixed'][dict_params['parameters'] == 'log_rho']= 1
initial_point,upper_bounds,lower_bounds,mu_prior,sigma_prior = get_params(dict_params) 

if dict_params['fixed'][dict_params['parameters'] == 'Omega'] == 0:
    print('Scaling spectrum to a proper Omega')
    Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type =  turn_params(initial_point,dict_params)
    pos,flux_syn_rank = syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type)
    new_omega = np.median(flux_obs_rank-flux_syn_rank)+omega
    dict_params['values'][dict_params['parameters'] == 'Omega'] = new_omega
    initial_point,upper_bounds,lower_bounds,mu_prior,sigma_prior = get_params(dict_params)


if method == 'MCMC':

    print('Preminimization')
    log_sigma_state = dict_params['fixed'][dict_params['parameters'] == 'log_sigma']
    log_rho_state   = dict_params['fixed'][dict_params['parameters'] == 'log_rho']
 
    dict_params['fixed'][dict_params['parameters'] == 'log_sigma']= 1
    dict_params['fixed'][dict_params['parameters'] == 'log_rho']= 1
    
    initial_point,upper_bounds,lower_bounds,mu_prior,sigma_prior = get_params(dict_params)                                                               
    par,cov_par,err_fact = do_curve_fit(initial_point,masked_flux_obs,masked_eflux_obs,masked_wave_obs,tol=0.001)

    epar = np.sqrt(np.diag(cov_par)) 
    param_result = par
    masked_eflux_obs = err_fact*masked_eflux_obs
    
    if error_map == 'True':
        print('Generating error map')
        centers,sigmas,amplitudes,sig_res = wide_errors(param_result,dict_params,flux_obs_rank,eflux_obs_rank,resol_wave,threshold) 
        extra_err_rank,extra_err = do_error_map(centers,sigmas,amplitudes,sig_res)
        err_tot = extra_err
        initial_point,upper_bounds,lower_bounds,mu_prior,sigma_prior = get_params(dict_params)
        par,cov_par,err_fact = do_curve_fit(initial_point,masked_flux_obs,masked_eflux_obs,masked_wave_obs,tol=0.001)
        err_tot = err_fact*err_tot
        epar = np.sqrt(np.diag(cov_par)) 
        param_result = par
    else:
        err_tot = err_fact*masked_eflux_obs
     
    eflux_obs_rank = err_fact*eflux_obs_rank 
    print("Errors underestimated by a factor of ",np.round(err_fact)) 
    print(print_result(param_result,dict_params))
    full_params  = dict_params['values']  
    fixed_params = dict_params['fixed']
    full_params[fixed_params < 1] = param_result
    dict_params['values'] = full_params
    dict_params['fixed'][dict_params['parameters'] == 'log_sigma'] =  log_sigma_state
    dict_params['fixed'][dict_params['parameters'] == 'log_rho']   =  log_rho_state

  
    param_result,upper_bounds,lower_bounds,mu_prior,sigma_prior = get_params(dict_params)

    print('Setting_MCMC')
    
    svec = 0.01*np.ones(len(param_result))
    for i in range(len(epar)):
        svec[i] = np.max([svec[i],epar[i]])

    ndim     = len(param_result)
    nwalkers = nwalk_fact*ndim
    with Pool() as pool:
        chain,lnprob,accrate  =  domcmc(param_result,nwalkers,nburn,nsteps,ndim,svec,pool)

    best_fit = np.zeros(ndim)
    sel_best = lnprob == np.max(lnprob)
    best_fit = np.array(chain[sel_best,:][0])
    print('Writting binary to file')
    fichr=open("BINOUT/"+name_star+"b.bin","wb")
    pic.dump(best_fit,fichr)
    pic.dump(chain,fichr)
    pic.dump(lnprob,fichr)
    pic.dump(accrate,fichr)
    pic.dump(par,fichr)
    pic.dump(epar,fichr)
    fichr.close()
    eflux_obs_rank  = np.sqrt(eflux_obs_rank**2.+extra_err_rank**2.)

elif method == 'LM':
    
    dict_params['fixed'][dict_params['parameters'] == 'log_sigma']= 1
    dict_params['fixed'][dict_params['parameters'] == 'log_rho']= 1
    
    initial_point,upper_bounds,lower_bounds,mu_prior,sigma_prior = get_params(dict_params)                                                               
    par,cov_par,err_fact = do_curve_fit(initial_point,masked_flux_obs,masked_eflux_obs,masked_wave_obs,tol=0.001)
    epar = np.sqrt(np.diag(cov_par)) 
    param_result = par
    if error_map == 'True':
        print('Generating error map')
        centers,sigmas,amplitudes,sig_res = wide_errors(param_result,dict_params,flux_obs_rank,eflux_obs_rank,resol_wave,threshold) 
        extra_err_rank,extra_err = do_error_map(centers,sigmas,amplitudes,sig_res)
        err_tot = extra_err
        initial_point,upper_bounds,lower_bounds,mu_prior,sigma_prior = get_params(dict_params)
        par,cov_par,err_fact = do_curve_fit(initial_point,masked_flux_obs,masked_eflux_obs,masked_wave_obs,tol=0.001)
        epar = np.sqrt(np.diag(cov_par)) 
        param_result = par
    else:
        err_tot = masked_eflux_obs     
   
    print(print_result(param_result,dict_params))
    print("Printing parameter results to file!")
    strings = name_star
    best_fit = par
    for i in range(len(best_fit)):
        strings = strings+' '+str(best_fit[i])+' '+str(epar[i]) 
    system("echo "+strings+" >> results_LM")
    best_fit = par

Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,log_sigma,log_rho,vmac_type,vsini_type,resol_type =  turn_params(best_fit,dict_params)
pos,flux_syn_rank = syn_spec(Teff,logg,MH,RV,resolution,vsini,ldc,vmac,omega,A_v,vmac_type,vsini_type,resol_type)
print('I will write the best fit and the observed flux')
write_spec(wave_obs_rank, flux_obs_rank, eflux_obs_rank,name_star)
write_spec(wave_obs_rank, flux_syn_rank, np.zeros(len(wave_obs_rank)), name_star+'best_fit')
print('Dark overlord, I am done!')
