#Program to find HI parameters of the sources in a 3D radio image:


'''
export LC_ALL='en_US.utf8'
python3 full_mpi.py


Inputs:
Fits line image cube required.

Outputs:
Catalogs with HI parameters in SDC#2 format.
'''

#Importing libraries:
import os, sys, re, shutil
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs

#Estimating the starting time of this program:
import time
t0=time.time()

#Define the function for logging:
import logging

def log_set():
    from datetime import datetime
    logfile = 'logfile_mpi.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w', format='%(levelname)s : %(message)s', datefmt="%d/%b/%Y %H:%M:%S")
    logging.info('Creating logfile for the program on '+str(datetime.now())+'\n')

    console = logging.StreamHandler()

    console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)

#Start the logging:
log_set()


#Multi-node Multi-core processing setting:
from mpi4py import MPI
comm=MPI.COMM_WORLD
num_process=comm.Get_size()
rank=comm.Get_rank()


#Make function to delete the temporary files or directories generated during program execution:
cwd = os.getcwd()
temp_files = ['new.param','sub_cube','aa','aper.fits','masked_','m0_']
def delete_temp(temp_files):
    for x in temp_files:
        sub_f=[ f for f in os.listdir(cwd) if f.startswith(x) or f.endswith(x)]
        for f in sub_f:
            if os.path.isfile(f):
                os.remove(os.path.join(cwd, f))
            if os.path.exists(f):
                shutil.rmtree(os.path.join(cwd, f))
#Delete the temporary files if they already exist:
if rank ==0 :
   delete_temp(temp_files)


#Make function to select a list of files based on some pattern in their names:
import glob
def selectFiles(pattern):
    files = glob.glob(pattern)
    return files

#Make function to delete a list of files based on some pattern in their names:
import glob
def deleteFiles(pattern):
    files = glob.glob(pattern)
    for f in files:
        os.remove(f)


#To ignore Python WARNINGs:
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")



###########################################################################################
# 1st Step: Running the source finding algorithm on each sliced channel image of the cube #
###########################################################################################

#Open the fits image file and get the image data and header:
filename='/o9000/SDC2/data/sky_full_v2.fits'
hdu = fits.open(filename, mode='readonly')
h = hdu[0].header
img = hdu[0].data
hdu.close()
if rank==0:
   logging.info('Number of frequency channels in the line cube: %d\n'%img.shape[0])

#Source finding parameters:
sigma_detectionThreshold = 2.2
pixel_minDetectionArea = 2

#SExtractor configuration file:
sextractor_configFile='/o9000/SDC2/SHAO/default.sex'

#Writing the file having description of output parameters to be evaluated using SExtractor:
if rank==0:
   with open('new.param', 'w') as outpars_file:
        outpars_file.write(r"""#NUMBER                   Running object number  
################################################################################
FLUX_AUTO           Flux within a Kron-like elliptical aperture         [count]
ALPHA_J2000         Right ascension of barycenter (J2000)               [deg]
DELTA_J2000         Declination of barycenter (J2000)                   [deg]
A_WORLD             Profile RMS along major axis (world units)          [deg]
B_WORLD             Profile RMS along minor axis (world units)          [deg]
THETA_WORLD         Position angle (CCW/world-x)                        [deg]
################################################################################
""")

#Make function to run SExtractor for a single channel 2D image fits file:
def sourceFind(imFits2D):
    f= open("%s.sh"%imFits2D,"w")
    lines=r"""#! /bin/sh

sex %s  -c %s  -CATALOG_NAME %s.cat  -PARAMETERS_NAME  new.param  -DETECT_MINAREA %d  -DETECT_THRESH %f  -THRESH_TYPE RELATIVE  -GAIN 0.0  -DEBLEND_MINCONT 1.0  -PIXEL_SCALE 0.0  -BACK_TYPE AUTO  -BACK_VALUE 0.0  -BACK_SIZE 32  -BACKPHOTO_TYPE LOCAL  -CHECKIMAGE_TYPE APERTURES  -CHECKIMAGE_NAME %s.aper.fits 2> %s.log1
"""%(imFits2D, sextractor_configFile, imFits2D, pixel_minDetectionArea, sigma_detectionThreshold, imFits2D, imFits2D)

    f.write(lines)
    f.close()
    os.system('chmod 777 %s.sh'%imFits2D)
    os.system('./%s.sh'%imFits2D)

#Source finding using SExtractor after slicing the line image cube into individual channel 2D images:
logging.info('Slicing the image cube into individual channel images and running SExtractor on them parallelly.\n')
#delete_temp(['.cat','.fits.log1','.fits.sh'])
f_1=h['CRVAL3']          #in Hz
d_f=h['CDELT3']          #channel width in Hz
freq=[]
pro_arr = np.array_split(np.arange(img.shape[0]),num_process)
for i in pro_arr[rank]:
    f_i=(f_1 + i*d_f)     #in Hz
    freq.append(f_i)
    h['CRVAL3']=f_i
    print(i)
    hdu = fits.PrimaryHDU(data=img[i,:,:],header=h)
    hdu.writeto(str(round(f_i))+'_a.fits', overwrite=True)
    sourceFind(str(round(f_i))+'_a.fits')
    os.system('rm %s %s %s %s' % (str(round(f_i))+'_a.fits', \
    str(round(f_i))+'_a.fits.log1',\
    str(round(f_i))+'_a.fits.sh',\
    str(round(f_i))+'_a.fits.aper.fits',))

comm.Barrier()

# Gather freq from all processes 
freq_ga = comm.gather(freq,root=0)
#freq_ga_new = np.concatenate(freq_ga,axis = 0)
print(freq_ga,"....gather freq")
if rank==0:
   freq_ga_new = np.concatenate(freq_ga,axis = 0)
   freq_comb = np.sort(freq_ga_new)
   print(freq_comb,"....shape=%d"%freq_comb.shape,"....rank=0")
# Bcast freq_comb to all processes
freq_comb = comm.bcast(freq_comb if rank == 0 else None, root=0)
print(freq_comb,"rank=%d"%rank)
########################################################################################
# 2nd Step: Cross-matching each single channel catalog with its next channel catalogs  #
########################################################################################

#List of all single-channel source-finding catalog files:
files0=[ f for f in os.listdir(cwd) if f.endswith('_a.fits.cat') ]
#sort a list of files having numbers in their names:
import natsort
files1=natsort.natsorted(files0,reverse=False)
files2=[ i.split('_a.fits.cat')[0]+'_b.cat' for i in files1 ]
files3=[ i.split('_a.fits.cat')[0]+'_c.cat' for i in files1 ]
files4=[ i.split('_a.fits.cat')[0]+'_d.cat' for i in files1 ]
files5=[ i.split('_a.fits.cat')[0]+'_e.cat' for i in files1 ]
files6=[ i.split('_a.fits.cat')[0]+'_f.cat' for i in files1 ]
files7=[ i.split('_a.fits.cat')[0]+'_g.cat' for i in files1 ]
print(files1)

#Extracting only RA and DEC from the channel catalog files:
if rank ==0 :
   for i in range(len(files1)):
       lines = open(files1[i], 'r').readlines()
       with open(files2[i], 'w') as outfile:
           #There may be some catalog files with no source detected in the frame.
           if lines[6:] == []:
               outfile.write("0.0 0.0 0.0 0.0 0.0")
           else:
               outfile.writelines(lines[6:])
       #Writing only RA and DEC:
       with open(files2[i],'r') as oldfile, open(files3[i], 'w') as newfile:
           for line in oldfile:
               line=line.split()
               RA_deg=line[1]
               DEC_deg=line[2]
               newfile.write(RA_deg+' '+DEC_deg+'\n')

# synchronization before Cross-match
comm.Barrier()

#Cross-match each single channel catalog with its next channel catalogs:
logging.info('Cross-match each single channel catalog with its next channel catalogs parallelly.')
#Make function to cross-match a single i^th channel catalog with its next channel catalogs:
def looping_over_channels(i):
    #Taking search radius between consecutive channels to be 7 arcsec considering the redshift range and beam size of the line cube:
    os.system("topcat -stilts tmatch2 matcher=sky params=7 in1='%s' ifmt1=ascii in2='%s' ifmt2=ascii join=1and2 find=best omode=out out='%s' ofmt=ascii values1='col1 col2' values2='col1 col2' fixcols=none"%(files3[i],files3[i+1],files4[i]))
    os.system("topcat -stilts tmatch2 matcher=sky params=7 in1='%s' ifmt1=ascii in2='%s' ifmt2=ascii join=1and2 find=best omode=out out='%s' ofmt=ascii values1='col1 col2' values2='col1 col2' fixcols=none"%(files3[i],files3[i+2],files5[i]))
    lines1 = open(files4[i], 'r').readlines()
    lines2 = open(files5[i], 'r').readlines()
    #Detecting source in at least 2 channel images out of 3 consecutive channel images:
    if lines1[1:] != []:
        open(files6[i], 'w').writelines(lines1[1:])
    if lines2[1:] != []:
        open(files6[i], 'a').writelines(lines2[1:])

pro_arr = np.array_split(np.arange(len(freq_comb)-2),num_process)
for k in pro_arr[rank]:
#for k in range(len(freq)-2):
    lines = open(files3[k], 'r').readlines()
    #Removing channel catalog with no source:
    if lines[0:] == ['0.0 0.0\n']:
        continue
    else:
        looping_over_channels(k)

comm.Barrier()

#Writing the position and channel frequency of the cross-matched sources into a catalog:
if rank==0:
   if os.path.exists('raw_cat1.txt'):
      os.remove('raw_cat1.txt')

lines = []
pro_arr = np.array_split(np.arange(len(freq_comb)),num_process)
for k in pro_arr[rank]:
#for k in range(len(freq_comb)):
    #Removing cross-matched channel catalog with no source:
    if not os.path.exists(files6[k]):
        continue
    #Checking if the frequency of channel and the channel catalog name is matching: Not needed
    if freq_comb[k]!=float(files6[k].split('_f.cat')[0]):
        logging.info('The frequency %f is not matching with channel catalog file name!'%freq_comb[k])
        continue
    #Remove the duplicate source positions:
    df1=pd.read_csv(files6[k],sep='\s+', na_filter=False,header=None, skiprows=0,dtype='unicode')
    df2=df1.drop_duplicates(subset=[0,1],keep='first')
    df2[5]=freq_comb[k]
    df2.to_csv(files7[k],sep=' ', columns=[0,1,5], header=False, index=False)
    #Save all source positions to a file:
    lines.extend(open(files7[k], 'r').read().splitlines())

comm.Barrier()

lines_ga = comm.gather(lines,root=0)

if rank==0:
   lines_comb = np.concatenate(lines_ga,axis = 0)
   print(lines_comb.shape,"==================lines")
   with open('raw_cat1.txt', 'w') as f:   
        fc = os.linesep.join(lines_comb)
        f.write(fc)
   #open('raw_cat1.txt', 'a').writelines(lines_comb)
   delete_temp(['.cat'])
#print('\n\n')














