

You need to install the following python packages to run SteParSyn (e.g. using pip):

numpy
scipy
matplotlib
astropy
extinction
emcee
tqdm
celerite
george
pygtc


	
The repository contains the following programs:

	- SteParSyn.py:  the main code
	- convsyn.py:    convolution code
	- readstats.py:  reads the Markov chains
	- runone.sh:  it calls SteParSyn.py
	- runall.sh:  it calls runone.sh 
	- readone.sh: it calls readstats.py 
	- readall.sh: it calls readone.sh 
	
	  
To install the code: clone the repo, install the dependencies, create missing dirs ... :

	$  git clone https://github.com/hmtabernero/SteParSyn 
	$  pip3 install --user numpy scipy matplotlib astropy extinction emcee tqdm celerite george pygtc
	$  cd SteParSyn
	$  mkdir eSPECTRA BINOUT PLOTS GRIDS
	
Download the synthetic grid (Fe_MARCS.bin) from https://cloud.cab.inta-csic.es/index.php/s/ApSLw6XnDmnXkkM 

You should put it in the folder GRIDS

Give permissions to programs:

	$ chmod +x *sh


Relevant directories:

	- GRIDS: here you should put the grid of synthetic spectra 
	- MASKS: line masks files
	- RANGES: your range files
	- CONFIG: here you can control the priors and which parameters to fix, priors, etc
	- OPT:   option files (iterations, number of chains, ...)
	- SPECTRA: input spectra 
	- eSPECTRA: best fit spectra (in ascii)
	- BINOUT: pickle files containing the MCMC sampling
	- PLOTS: corner plots 

The repo contains an example to compute the stellar atmospheric parameters of a solar spectrum. To run the example:

	$ ./runall.sh

To get the results:

	$ ./readall.sh

If you want to analyse your own data you need to do as follows (I am assuming that the name of your star is "sun_vesta")

	- Provide a txt file containing: wavelengths, fluxes, and its uncertainities (same as SPECTRA/sun_vesta.txt)
	- Create a mask file with: line names, line centre, mask centre, and mask width (same format as MASKS/sun_vesta_masks.txt)
	- Generate an option file with your parameter options: limits, priors, etc. (You can use CONFIG/sun_vesta.conf as is)
	
Add the following line to runall.sh:

	./runone.sh sun_vesta hermes MCMC

Add also this line to readall.sh

	./readone.sh sun_vesta 
 


I will write a more complete manual in the near future. 

Finally, please cite the SteParSyn paper if you use it in your reasearch, you can find it here: https://ui.adsabs.harvard.edu/abs/2022A%26A...657A..66T/abstract

If you have any doubts please do not hesitate to e-mail me.






