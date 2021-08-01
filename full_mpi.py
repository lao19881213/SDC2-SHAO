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

#########################################################################
# 3th Step: Finding approximate source position and central frequency   #
#########################################################################

if rank==0:
   #Define the spatial boundary of each channel image:
   wcs_info=wcs.WCS(h).celestial
   minPixelCoordinate_deg=wcs_info.all_pix2world([[1,1]],1)
   RAmax = minPixelCoordinate_deg[0][0]
   DECmin = minPixelCoordinate_deg[0][1]
   maxPixelCoordinate_deg=wcs_info.all_pix2world([[h['NAXIS1'],h['NAXIS2']]],1)
   RAmin = maxPixelCoordinate_deg[0][0]
   DECmax = maxPixelCoordinate_deg[0][1]
   
   #Define the spectral boundary of the line cube:
   #fmin = min(freq)
   #fmax = max(freq)
   fmin=f_1
   fmax=f_1 + (h['NAXIS3']-1)*d_f
   
   #Find the approximate central position and range of channels for each source:
   logging.info('Finding approximate source position and central frequency.\n')
   Xcenter=[]
   Ycenter=[]
   freq_min=[]
   freq_max=[]
   freq_mid=[]
   shutil.copy('raw_cat1.txt','aa1')
   lines1 = open('aa1', 'r').readlines()
   i=len(lines1)
   for x in range(len(lines1)):
       lines = open('aa1', 'r').readlines()
       if len(lines)==0:
           os.remove('aa1')
           break
       else:
           line1=lines[0].split()
           RA1=float(line1[0])
           DEC1=float(line1[1])
           Freq1=float(line1[2])
           out=open('aa2.{}'.format(i), 'w')
           for line in lines:
               line2=line.split()
               RA2=float(line2[0])
               DEC2=float(line2[1])
               Freq2=float(line2[2])
               #Taking search radius between all detection channels to be 7 arcsec and total frequency width of the detection to be 15 channels with 30 kHz each channel width considering the small beam size of 7 arcsec and redshift coverage of frequency bandwidth:
               if (np.sqrt(((RA1-RA2)*np.cos(np.radians(DEC1),dtype=np.float128))**2 + (DEC1-DEC2)**2)*3.6E+3 <= 7) and (abs(Freq1-Freq2) < 15*abs(d_f)):
                   out.write(line)
           out.close()
           #Remove the lines from a file that are common in other file:
           lines2 = open('aa2.{}'.format(i), 'r').readlines()
           with open('aa1','r') as oldfile, open('aa1.{}'.format(i), 'w') as newfile:
               for y in oldfile:
                   if not y in lines2:
                       newfile.write(y)
           os.remove('aa1')
           shutil.copy('aa1.{}'.format(i),'aa1')
           #This will remove all the lines except the first having coordinate information for the same source in different channels in file aa2.
           #Getting rough estimates of source position and central frequency:
           df1=pd.read_csv('aa2.{}'.format(i),sep=' ', na_filter=False,header=None, skiprows=0,dtype='unicode')
           #Considering detection in at least 3 consecutive channels:
           #if len(df1) > 1:
           X1=pd.to_numeric(df1[0]).mean()
           Y1=pd.to_numeric(df1[1]).mean()
           freq_min1=min(pd.to_numeric(df1[2]))-2*abs(d_f) #Adding 2 channel on lower side.
           #The maximum frequency emission channel should be maximum channel frequency + 2 channels, as we considered the first channel frequency while cross-matching the consecutive channels. But, as we considered one channel gap also in 3 consecutive channel emission, it should be as follows:
           freq_max1=max(pd.to_numeric(df1[2]))+4*abs(d_f) #Adding 2 channel on higher side.
           freq_mid1=pd.to_numeric(df1[2]).median()+abs(d_f)
           #Ignoring the sources on the edged 7 pixels of each channel image and 3 channels from both the edges of line cube:
           if X1>RAmin+7*abs(h['CDELT1']) and X1<RAmax-7*abs(h['CDELT1']) and Y1>DECmin+7*abs(h['CDELT2']) and Y1<DECmax-7*abs(h['CDELT2']) and freq_min1>fmin and freq_max1<fmax:
               Xcenter.append(X1)
               Ycenter.append(Y1)
               freq_min.append(freq_min1)
               freq_max.append(freq_max1)
               freq_mid.append(freq_mid1)
           i=i-len(lines2)
   
   #print(Xcenter,Ycenter,freq_min,freq_max,freq_mid)
   with open('raw_cat2.txt', 'w') as outfile:
       for i in range(len(Xcenter)):
           outfile.write(str(Xcenter[i])+' '+str(Ycenter[i])+' '+str(freq_min[i])+' '+str(freq_max[i])+' '+str(freq_mid[i])+'\n')
   
   if len(Xcenter)==0:
       logging.info('No HI source found in the line cube! Check the input parameters!!')
       sys.exit()
   else:
       logging.info('Number of HI sources found = %d\n'%len(Xcenter))

comm.Barrier()

Xcenter = comm.bcast(Xcenter if rank == 0 else None, root=0)
Ycenter = comm.bcast(Ycenter if rank == 0 else None, root=0)
freq_min = comm.bcast(freq_min if rank == 0 else None, root=0)
freq_max = comm.bcast(freq_max if rank == 0 else None, root=0)
freq_mid = comm.bcast(freq_mid if rank == 0 else None, root=0)

###################################################
# 4th Step: Making moment-0 map for each source   #
###################################################

#Extract a spatial and spectral subcube using the spectral_cube python module:
#Slicing the full-size image cube into smaller image cubes of 14 pixel = 2.8*14 = 39.2 arcsec spatial length around the HI sources within its channel range of HI emission:
logging.info('Slicing the full size image cube into sub-cubes around the HI sources parallelly.\n')
from spectral_cube import SpectralCube
import astropy.units as u
cube = SpectralCube.read(filename)
pro_arr = np.array_split(np.arange(len(Xcenter)),num_process)
for i in pro_arr[rank]:
#for i in range(len(Xcenter)):
    #print(i)
    #Note that the spatial length along RA and DEC axes are different.
    sub_cube_i = cube.subcube(xlo=(Xcenter[i]-abs(h['CDELT1'])*7)*u.deg, xhi=(Xcenter[i]+abs(h['CDELT1'])*7)*u.deg, ylo=(Ycenter[i]-abs(h['CDELT2'])*6)*u.deg, yhi=(Ycenter[i]+abs(h['CDELT2'])*7)*u.deg, zlo=freq_min[i]*u.Hz, zhi=freq_max[i]*u.Hz)
    sub_cube_i.write('sub_cube_{}.fits'.format(i))

#Make function to estimate rms (in units of Jy/beam) of a subcube:
def get_rmsCube(cubeFile):
    #Read the cube fits file:
    hdu = fits.open(cubeFile, mode='readonly')
    h = hdu[0].header
    img = hdu[0].data
    hdu.close()
    #Number of frequency channels in the cube:
    nchan=h['NAXIS3']
    rmsChans=[]
    #estimating median absolute deviation for each channel of cube:
    for i in range(nchan):
        im2D_i = img[i,:,:]
        MAD_i = 1.4826*np.nanmedian(abs(im2D_i-np.nanmedian(im2D_i)))
        rmsChans.append(MAD_i)
    rms=np.median(np.array(rmsChans))
    return rms

#The above function is better than the 'astropy.stats.mad_std' function, because the function 'get_rmsCube' takes the median of rms values for individual channels to take care of presence of any source/signal in any channel.

comm.Barrier()

#Make moment-0 map corresponding to each subcubes:
logging.info('Making moment-0 map for each HI source.\n')
subcubes=[ c for c in os.listdir(cwd) if c.startswith('sub_cube_')]
rms_subcubes=[]
pro_arr = np.array_split(np.arange(len(subcubes)),num_process)
for i in pro_arr[rank]:
#for subcube in subcubes:
    #Read the spectral cube fits file:
    subcube1 = SpectralCube.read(subcubes[i])
    #Estimate the rms of the subcube:
    rms_1 = subcube1.mad_std(axis=None, how='cube').value
    rms_2 = get_rmsCube(subcubes[i])
    rms_subcube=min(rms_1,rms_2)
    #print(rms_1,rms_2,rms_subcube)
    rms_subcubes.append(rms_subcube)
    #Mask the subcube below 2.5 sigma level:
    #masked_cube1=subcube1.with_mask(subcube1 > 2.5*rms_subcube * u.Jy / u.beam)
    #Mask the subcube for negative values and write to fits file:
    masked_cube1=subcube1.with_mask(subcube1 > 0.0 * u.Jy / u.beam)
    masked_cube1.write('masked_'+subcubes[i],overwrite=True)
    #Compute the 0th order moment and write to file:
    moment0_map = masked_cube1[1:-2,:,:].moment(order=0, axis=0, how='cube')
    moment0_map.write('m0_'+subcubes[i],overwrite=True)

####################################################
# 5th Step: Source finding from the moment-0 maps  #
####################################################
comm.Barrier()
#Running SExtractor on the moment-0 maps:
logging.info('Source finding for all moment-0 images using SExtractor parallelly.\n')
sigma_detectionThreshold = 2.0
pixel_minDetectionArea = 3
files_11=[ f for f in os.listdir(cwd) if f.startswith('m0_') ]
files_12=natsort.natsorted(files_11,reverse=False)
comm.Barrier()
pro_arr = np.array_split(np.arange(len(files_12)),num_process)
for i in pro_arr[rank]:
    imFits2D=files_12[i]
    sourceFind(imFits2D)
    os.system('rm %s %s %s' % (imFits2D +'.log1',\
    imFits2D +'.sh',\
    imFits2D +'.aper.fits'))

comm.Barrier()

#if rank==0:
#   delete_temp(['.fits.log1','.fits.sh'])
#   deleteFiles('m0_sub_cube_*.fits.aper.fits')

#Make function to estimate rms and background level (in units of Jy/beam) of a 2D image:
def get_rmsBkg(imFits2D):
    hdu = fits.open(imFits2D, mode='readonly')
    h = hdu[0].header
    img = hdu[0].data
    hdu.close()
    rms = 1.4826*np.nanmedian(abs(img-np.nanmedian(img)))
    bkg = np.nanmedian(img)
    return rms,bkg

#Make function for source finding in moment-0 map in case of significant background gradient:
def m0_sourceFind(imFits2D,sigma_detectionThreshold=1.1):
    #os.remove('%s.cat'%imFits2D)
    deleteFiles('%s*.log1*'%imFits2D)
    deleteFiles('%s*.sh*'%imFits2D)
    deleteFiles('%s*.1cat*'%imFits2D)
    rms,bkg = get_rmsBkg(imFits2D)
    f= open("%s.sh"%imFits2D,"w")
    lines=r"""#! /bin/sh

sex %s  -c %s  -CATALOG_NAME %s.1cat  -PARAMETERS_NAME  new.param  -DETECT_MINAREA %d  -DETECT_THRESH %f  -THRESH_TYPE ABSOLUTE  -GAIN 0.0  -DEBLEND_MINCONT 1.0  -PIXEL_SCALE 0.0  -BACK_TYPE MANUAL  -BACK_VALUE %f  -BACK_SIZE 32  -BACKPHOTO_TYPE LOCAL  -CHECKIMAGE_TYPE NONE 2> %s.log1
"""%(imFits2D, sextractor_configFile, imFits2D, pixel_minDetectionArea, sigma_detectionThreshold*rms, rms, imFits2D)

    f.write(lines)
    f.close()
    os.system('chmod 777 %s.sh'%imFits2D)
    os.system('./%s.sh'%imFits2D)
    deleteFiles('%s*.log1*'%imFits2D)
    deleteFiles('%s*.sh*'%imFits2D)


#Make function to get the mean RA and DEC in a moment-0 map:
def mean_RA_DEC(m0_filename):
    hdu = fits.open(m0_filename, mode='readonly')
    h = hdu[0].header
    img = hdu[0].data
    hdu.close()
    wcs_info=wcs.WCS(h).celestial
    meanCoordinate_deg=wcs_info.all_pix2world([[h['NAXIS1']/2,h['NAXIS2']/2]],1)
    RAmean = meanCoordinate_deg[0][0]
    DECmean = meanCoordinate_deg[0][1]
    return RAmean,DECmean

#List of SExtractor catalog files:
files_1=[ f for f in os.listdir(cwd) if f.endswith('.fits.cat') ]
files_2=natsort.natsorted(files_1,reverse=False)

#Flux measured by SExtractor does not take care of the beam info of the radio fits image. We need therefore to convert from the flux density unit in the maps (Jy/beam) to the units required for the integrated flux (Jy) by dividing the measured flux with the number of pixels in the beam.
Npix_beam = (1.133 * h['BMAJ'] * h['BMIN']) / (abs(h['CDELT1']) * abs(h['CDELT2']))
logging.info('Number of pixels in the beam area = %f\n'%Npix_beam)

#Read the SExtractor output catalog file in python:
from astropy.io import ascii
#files_3=[ i.split('.fits.cat')[0]+'.txt' for i in files_2 ]
#delete_temp(files_3)
#results_m0 = []
pro_arr = np.array_split(np.arange(len(files_2)),num_process)
for fn in pro_arr[rank]:
#for fileID,fileName in enumerate(files_2):
    fileName = files_2[fn]
    Data = ascii.read(fileName)
    #print(Data)
    #If no source is found in moment-0 map for the current parameters, we use different parameter setup.
    if len(Data['ALPHA_J2000'])==0:
        #print(fileID)
        fitsName=fileName.split('.cat')[0]
        m0_sourceFind(fitsName)
        Data = ascii.read(fitsName+'.1cat')
        if len(Data['ALPHA_J2000'])==0:
            m0_sourceFind(fitsName,sigma_detectionThreshold=1.0)
            Data = ascii.read(fitsName+'.1cat')
        #Selecting the source nearest to center in case of many detections:
        RAmean,DECmean=mean_RA_DEC(fitsName)
        dist=[]
        for i in range(len(Data['ALPHA_J2000'])):
            dist_i=np.sqrt(((RAmean-Data['ALPHA_J2000'][i])*np.cos(np.radians(DECmean),dtype=np.float128))**2 + (DECmean-Data['DELTA_J2000'][i])**2)
            dist.append(dist_i)
        #print(np.argmin(dist))
        if len(dist)!=0:
           RA_deg=Data['ALPHA_J2000'][np.argmin(dist)]
           DEC_deg=Data['DELTA_J2000'][np.argmin(dist)]
           flux_JyHz=Data['FLUX_AUTO'][np.argmin(dist)]/Npix_beam
           A_arcsec=np.round(Data['A_WORLD'][np.argmin(dist)]*3600*4,3)
           B_arcsec=np.round(Data['B_WORLD'][np.argmin(dist)]*3600*4,3)
           PA_deg=np.round(Data['THETA_WORLD'][np.argmin(dist)],3)
        os.remove(fitsName+'.1cat')
    elif len(Data['ALPHA_J2000'])==1:
        RA_deg=Data['ALPHA_J2000'][0]
        DEC_deg=Data['DELTA_J2000'][0]
        flux_JyHz=Data['FLUX_AUTO'][0]/Npix_beam
        A_arcsec=np.round(Data['A_WORLD'][0]*3600*4,3)
        B_arcsec=np.round(Data['B_WORLD'][0]*3600*4,3)
        PA_deg=np.round(Data['THETA_WORLD'][0],3)
    #If more than one source is found in moment-0 map, we select the nearest to the center.
    #elif len(Data['ALPHA_J2000'])>1:
    else:
        #print(fileID)
        RAmean,DECmean=mean_RA_DEC(fileName.split('.cat')[0])
        dist=[]
        for i in range(len(Data['ALPHA_J2000'])):
            dist_i=np.sqrt(((RAmean-Data['ALPHA_J2000'][i])*np.cos(np.radians(DECmean),dtype=np.float128))**2 + (DECmean-Data['DELTA_J2000'][i])**2)
            dist.append(dist_i)
        #print(np.argmin(dist))
        if len(dist)!=0:
           RA_deg=Data['ALPHA_J2000'][np.argmin(dist)]
           DEC_deg=Data['DELTA_J2000'][np.argmin(dist)]
           flux_JyHz=Data['FLUX_AUTO'][np.argmin(dist)]/Npix_beam
           A_arcsec=np.round(Data['A_WORLD'][np.argmin(dist)]*3600*4,3)
           B_arcsec=np.round(Data['B_WORLD'][np.argmin(dist)]*3600*4,3)
           PA_deg=np.round(Data['THETA_WORLD'][np.argmin(dist)],3)
    #Inclination angle estimation:
    i_deg=np.degrees(np.arccos(np.sqrt(((B_arcsec/A_arcsec)**2 - 0.04)/0.96)))
    #results_m0.append('%s %s %s %s %s %s\n' % (str(RA_deg), str(DEC_deg), \
    #str(flux_JyHz), str(A_arcsec), str(PA_deg), str(i_deg)))
    with open(fileName.split('.fits.cat')[0]+'.txt', 'w') as outfile:
        outfile.write(str(RA_deg)+' '+str(DEC_deg)+' '+str(flux_JyHz)+' '+str(A_arcsec)+' '+str(PA_deg)+' '+str(i_deg)+'\n')

if rank==0:
   delete_temp(files_1)

comm.Barrier()
#################################################################################
# 6th Step: Estimating central frequency and line-width from global HI profile  #
#################################################################################
if rank==0:
   #List of catalog and masked subcube files:
   files_4=[ i.split('.fits.cat')[0] for i in files_2 ]
   files_5=[ 'masked_'+i.split('m0_')[1]+'.fits' for i in files_4 ]
   
   #Make function to estimate global HI profile of a masked subcube:
   def get_HIprofile(cubeFile,RA_deg,DEC_deg,A_arcsec):
       #Read the cube fits file:
       hdu = fits.open(cubeFile, mode='readonly')
       h = hdu[0].header
       img = hdu[0].data
       hdu.close()
       #Number of frequency channels in the cube:
       nchan=h['NAXIS3']
       #print(nchan)
       #Pixel coordinates of the source center:
       wcs_info=wcs.WCS(h).celestial
       center_pix=wcs_info.all_world2pix(RA_deg,DEC_deg,1)
       X_pix=round(float(center_pix[0]))
       Y_pix=round(float(center_pix[1]))
       #print(X_pix,Y_pix)
       #Estimating frequencies of each channel and flux within source extent in each channel:
       freq_Hz=[]
       flux_mJy=[]
       for i in range(nchan):
           f_i=round((h['CRVAL3'] + (-h['CRPIX3']+1+i)*h['CDELT3'])/10000)     #in 10000 Hz unit
           freq_Hz.append(f_i)
           img1=img[i,:,:]
           region = img1[(Y_pix-round(A_arcsec/abs(2*h['CDELT1']*3.6E3))-1):(Y_pix+round(A_arcsec/abs(2*h['CDELT1']*3.6E3))+1),(X_pix-round(A_arcsec/abs(2*h['CDELT1']*3.6E3))-1):(X_pix+round(A_arcsec/abs(2*h['CDELT1']*3.6E3))+1)]
           flux_mJy_i = np.nansum(region)*1000/Npix_beam
           flux_mJy.append(flux_mJy_i)
       freq_Hz=np.array(freq_Hz)
       flux_mJy=np.array(flux_mJy)
       return freq_Hz,flux_mJy
   
   
   #Make function to fit Gaussian model to the global HI profile:
   spatial_resolution=7     #arcsec
   from scipy.optimize import curve_fit
   def fit_HIprofile(cubeFile):
       #Read the catalog estimated from moment-0 map:
       m0_Cat='m0_'+cubeFile.split('masked_')[1]
       m0_Cat=m0_Cat.split('.fits')[0]+'.txt'
       line=open(m0_Cat, "r").readlines()[0]
       #print(line)
       RA_deg=float(line.split()[0])
       DEC_deg=float(line.split()[1])
       flux_JyHz=float(line.split()[2])
       A_arcsec=float(line.split()[3])          #semi-major axis length in arcsec
       #a_arcsec=np.sqrt((2*A_arcsec)**2 - spatial_resolution**2)   #deconvolved source size
       a_arcsec=A_arcsec                        #Need to check
       pa_deg=float(line.split()[4])+90.0       #position angle measured from +y axis counter-clockwise.
       if pa_deg<0.0:
           pa_deg=360+pa_deg
       i_deg=float(line.split()[5])
       #print(RA_deg,DEC_deg,flux_JyHz,A_arcsec,PA_deg,i_deg)
       #Find the global HI profile for the masked line cube:
       freq_Hz,flux_mJy=get_HIprofile(cubeFile,RA_deg,DEC_deg,A_arcsec)
       try:
           #Function for single Gaussian having amplitude, mean and standard deviation plus a continuum offset parameters:
           def gaussian(x, amp,mu,sigma,offset=min(flux_mJy)):
               return amp*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-mu)/sigma)**2))) + offset
           #Initial guesses for the parameters to fit gaussian function:
           amp0=max(flux_mJy)
           mu0=np.mean(freq_Hz)
           sigma0=(np.median(freq_Hz)-min(freq_Hz)-2*d_f/10000)/2
           pint=[amp0,mu0,sigma0]
           #print(pint)
           #Fitting:
           p_fit, pcov = curve_fit(gaussian, freq_Hz, flux_mJy, p0=pint, sigma=None, absolute_sigma=False, method='lm', maxfev=10000*(len(freq_Hz)+1), full_output=False)
           perr = np.sqrt(np.diag(pcov))
           #print(p_fit)
           mu=p_fit[1]
           freq_central=mu*1E4          #in Hz
           sigma=max(p_fit[2],sigma0)
           sigma=min(sigma,(max(freq_Hz)-min(freq_Hz)-4*d_f/10000)/2)
           w20=2*sigma*np.sqrt(2*np.log(5))*1E4     #in Hz
           #print(w20)
       except:
           logging.info('Fitting of HI profile failed for %s'%cubeFile)
           freq_central=np.mean(freq_Hz)
           w20=(freq_Hz[-3]-freq_Hz[2])*1E4     #in Hz
       c=299792.458             #in km/s
       freq_rest=1420405752.0   #in Hz
       W20=w20*c*freq_rest/(freq_central**2)    #in km/s
       #print(W20)
       return freq_central,W20,RA_deg,DEC_deg,flux_JyHz,a_arcsec,pa_deg,i_deg
   
   #Estimating central frequency and line-width for all HI source:
   logging.info('Estimating central frequency and line-width for all HI source global profiles.\n')
   freq_central=[];W20=[];RA_deg=[];DEC_deg=[];flux_JyHz=[];a_arcsec=[];pa_deg=[];i_deg=[]
   for i in range(len(files_4)):
       if os.path.exists(files_4[i]+'.txt'):
           freq_central_i,W20_i,RA_deg_i,DEC_deg_i,flux_JyHz_i,a_arcsec_i,pa_deg_i,i_deg_i=fit_HIprofile(files_5[i])
           #print(freq_central_i,W20_i)
           freq_central.append(freq_central_i)
           W20.append(W20_i)
           RA_deg.append(RA_deg_i)
           DEC_deg.append(DEC_deg_i)
           flux_JyHz.append(flux_JyHz_i)
           a_arcsec.append(a_arcsec_i)
           pa_deg.append(pa_deg_i)
           i_deg.append(i_deg_i)
   
   deleteFiles('m0_sub_cube_*.txt')



#############################################################
# Last Step: Writing the final catalog with HI parameters   #
#############################################################

   #Making the catalog:
   logging.info('Making the final catalog with HI parameters.\n')
   with open('SHAO_Team_full.txt', 'w') as outfile:
       outfile.write('id ra dec hi_size line_flux_integral central_freq pa i w20\n')
       for i in range(len(RA_deg)):
           outfile.write('%d %f %f %f %f %f %f %f %f\n'%(i,RA_deg[i],DEC_deg[i],a_arcsec[i],flux_JyHz[i],freq_central[i],pa_deg[i],i_deg[i],W20[i]))
   
   
   #Delete the temporary files:
   temp_files=['new.param','sub_cube','aa','aper.fits','masked_','m0_']
   delete_temp(temp_files)
   
   #Estimating the ending time of this program:
   t1=time.time()
   if (t1-t0) <= 60.0:
       run_time = str(np.round(t1-t0,2)) + ' seconds'
   elif 3600.0 >= (t1-t0) > 60.0:
       run_time = str(np.round((t1-t0)/60.0,2)) + ' minutes'
   else:
       run_time = str(np.round((t1-t0)/3600.0,2)) + ' hours'
   logging.info('********** Program Completed **********')
   logging.info('Time taken by the program = %s' % run_time)
















