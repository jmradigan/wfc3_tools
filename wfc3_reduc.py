import datetime, glob, os, pyfits, shutil, sys, pdb

import numpy as np
import matplotlib.pyplot as plt

import pyraf
from pyraf import iraf
from iraf import stsdas, analysis, slitless, axe, images, imutil

from stsci.tools import teal
from astropy.io import fits
from astropy.visualization import PercentileInterval
from astropy.io import ascii

import drizzlepac
from drizzlepac import astrodrizzle

from embed_subs import embed_subs

def wfc3_zeropoint_mags():
  mag_zeropoint = {  "F105W": 26.2687,
                     "F110W": 26.8223,
                     "F125W": 26.2303,
                     "F140W": 26.4524,
                     "F160W": 25.9463,
                     "F098M": 25.6674,
                     "F127M": 24.6412,
                     "F139M": 24.4793,
                     "F153M": 24.4635,
                     "F126N": 22.8609,
                     "F128N": 22.9726,
                     "F130N": 22.9900,
                     "F132N": 22.9472,
                     "F164N": 22.9089,
                     "F167N": 22.9568
                 }
  return(mag_zeropoint)

def wfc3_photplam():
  photplam = {   "F105W": 10552.,
                 "F110W": 11534.,
                 "F125W": 12486.,
                 "F140W": 13923.,
                 "F160W": 15369.,
                 "F098M": 9684.1,
                 "F127M": 12740.,
                 "F139M": 13838.,
                 "F153M": 15322.,
                 "F126N": 12585.,
                 "F128N": 12832.,
                 "F130N": 13006.,
                 "F132N": 13188.,
                 "F164N": 16403.,
                 "F167N": 16642.
             }
  return(photplam)
    
def embed_subarrays(datapath,fullframepath):
  """
  Embeds subarray images in a full frame image for the aXe pipeline.
  Uses the embed_subs script from STScI.

  INPUTS
  -------
  datapath : string
    path to subarry data

  fullframepath :string
    path where full frame data will be stored
    will be created if does not exist
  """
  inpath = datapath
  outpath = fullframepath

  #If the fullframe path doesn't exist, then create it
  if not os.path.exists(fullframepath):
    os.makedirs(fullframepath)

  #Get a list of all q _flt.fits files in the datapath
  files = glob.glob(os.path.join(inpath,'*q_flt.fits'))
  
  #Copy original _flt and _spt files to fullframe directory to avoid modification
  #These files are needed for the embed_subs script
  # (But only do this if the fullframe files don't already exist)
  for file in files:
    fullframe_filename=file.replace("q_flt","f_flt")
    spt_file=file.replace("_flt","_spt")
    if not os.path.exists(os.path.join(outpath,fullframe_filename)):
      shutil.copy(file,os.path.join(outpath,os.path.split(file)[1]))
      shutil.copy(spt_file,os.path.join(outpath,os.path.split(spt_file)[1]))

  #Create full frame images for all *q_flt.fits files in fullframe directory
  subfiles = glob.glob(os.path.join(outpath,'*q_flt.fits'))
  embed_subs(subfiles)        

  #Now go ahead and delete the copies of the original _flt and _spt files
  copies = glob.glob(os.path.join(outpath,"*q_flt.fits")) + glob.glob(os.path.join(outpath,"*spt.fits"))
  for file in copies:
    os.remove(file)
    
def filter_list(fullframepath,imdrizzlepath):
  """
    Returns a dictionary containig a file list for each filter/grism in the dataset

    INPUTS
    -------
    fullframepath : string
      path to full frame data

    imdrizzlepath :string
      path where image files will be copied to
      will be created if does not exist

  """

  #Get a list of all fullframe files
  files = glob.glob(os.path.join(fullframepath,"*f_flt.fits"))

  #Dictionary to store file lists
  filters = {}

  #Populate filter dictionary with file lists
  for file in files:
    fits = pyfits.open(file)
    filter = fits[0].header["FILTER"]
    fits.close()
    if filter not in filters:
      filters[filter] = []
    filters[filter].append(file)

  #For each filter write the file list to a text file with extension .lis
  #Place images into a directory to be drizzled (create directory if it doesn't exist)
  if not os.path.exists(imdrizzlepath):
    os.makedirs(imdrizzlepath)

  for filter in filters:
    f = open(os.path.join(fullframepath,filter+".lis"),'w') #Create a list file for that filter

    for file in filters[filter]:
      (path,fname) = os.path.split(file)
      f.write(fname+"\n")
      if filter[0].lower() == 'f': #This is an image, not a spectrum
        shutil.copy(file,os.path.join(imdrizzlepath,fname)) #copy to imdrizzlepath
    f.close()
    if filter[0].lower() == 'f': #If this is an image, then copy the list file to imdrizzlepath
      shutil.copy(os.path.join(fullframepath,filter+'.lis'),os.path.join(imdrizzlepath,filter+'.lis'))
  return filters

def do_astrodrizzle(filters,imdrizzlepath):
  """
    Runs astrodrizzle in order to stack all images.  Writes output as <filter>_drz.fits
    Note that we haven't dithered; the stack is just a higher S/N image of our FoV

    INPUTS:
    --------
    filters : dictionary {filter : filelist}

    imdrizzle path : string
      path where full frame images have ben stored for stacking
      
  """
  teal.unlearn('astrodrizzle') #teal mimics IRAF functionality, unlearn sets all parameters to default
  for filter in filters:
    if filter[0].lower() == 'f': #Check that we only have images
      if not os.path.exists(os.path.join(imdrizzlepath,filter+'_drz.fits')):
        old_dir = os.getcwd()
        os.chdir(imdrizzlepath)  #do the stacking in the imdrizzle directory
        astrodrizzle.AstroDrizzle(filters[filter],output=filter,build='yes',skysub=False)
        os.chdir(old_dir)  #switch back to original working directory


def do_sextractor(filters,imdrizzlepath):
  """
    Runs Source Extractor (SExtractor - can we all agree this is a bad name?)
    to identify target and reference stars
    Uses IRAF/pyRAF to get things in appropriate format (*barfing emoji* - oh well)

    INPUTS:
    --------
    filters : dictionary {filter : filelist}

    imdrizzle path : string
      path where full frame images have drizzled together
      
  """
  working_dir = os.getcwd()
  for filter in filters:
    if filter[0].lower() == 'f': #This is an image, not a spectrum

      #check if catalog already exists, if not run SExtractor
      if not os.path.isfile(os.path.join(imdrizzlepath,filter+'.cat')):

        #SExtractor needs the SCI and WHT extensions as separate fits files
        #IRAF fails if these separate images already exist, so clean up first
        try:
          os.remove(os.path.join(imdrizzlepath,'%s_drz_sci.fits'%(filter)))
        except OSError:
          pass
        try:
          os.remove(os.path.join(imdrizzlepath,'%s_drz_wht.fits'%(filter)))
        except OSError:
          pass

        #Using IRAF to copy the file extensions we want into their own images
        #(We are using IRAF instead of astropy to ensure that files are in correct format for SExtractor)
        iraf.imcopy(os.path.join(imdrizzlepath,'%s_drz.fits[SCI]'%(filter)),os.path.join(imdrizzlepath,'%s_drz_sci.fits'%(filter)))
        iraf.imcopy(os.path.join(imdrizzlepath,'%s_drz.fits[WHT]'%(filter)),os.path.join(imdrizzlepath,'%s_drz_wht.fits'%(filter))) 

        #Look for SExtractor default files in working directory, and copy to imdrizzlepath
        default_files=['default.sex','default.param','default.conv']
        missing_default_files = [df for df in default_files if not os.path.isfile(os.path.join(imdrizzlepath,df))];
        if len(missing_default_files) > 0:
          for df in missing_default_files:
            try:
              shutil.copy(os.path.join(working_dir,df),os.path.join(imdrizzlepath,df))
            except IOError, e:
              print "Unable to copy file. %s" % e

        #Run SExtractor in imdrizzlepath, then change back to working_dir
        #SExtractor creates the .cat file with source positions
        os.chdir(imdrizzlepath)
        zp = wfc3_zeropoint_mags()[filter]
        os.system('sex -c default.sex -WEIGHT_IMAGE %s_drz_wht.fits -MAG_ZEROPOINT %f -CATALOG_NAME %s.cat %s_drz_sci.fits' % (filter,zp,filter,filter)) 
        os.chdir(working_dir)

        #Change the column heading MAG_AUTO to MAG<filter pivot wavlength> to prepare catalog for aXe
        #This is instructed on pg 13 of the WFC3 IR grism handbook by J. Lee.
        pw=wfc3_photplam()[filter]
        cat = open('%s.cat'%(filter),'r')
         out_cat = open('%s_prep.cat'%(filter),'w')
           for line in f.readlines():
             if 'MAG_AUTO' in line:
               line = line.replace('MAG_AUTO','MAG_F%f'%(pw))
             out_cat.write(line)
          out_cat.close()
          f.close()
        
def identify_target(filter,imdrizzlepath):
  """

    Identify target amoung sources catalogued by Source Extractor.

    INPUTS:
    --------
    filter : string
      filter of the drizzled image

    imdrizzle path : string
      path where full frame drizzled image is located

    target_name : string
      optionally provide a target name,  default is 'Target'
      the output catalog is saved using this name
      
  """
  #open the drizzled image file for given filter
  imfile = os.path.join(imdrizzlepath,filter+"_drz.fits")
  try:
    hdu_list = fits.open(imfile)
  except IOError:
    print('The following file does not exist, or could not be opened: %s'(imfile))
    sys.exit()

  #pull image from hdu_list
  image_data = hdu_list['SCI'].data #2D numpy array

  #pull weights from hdu_list
  wht_data = hdu_list['WHT'].data #2D numpy array

  #close the fits file
  hdu_list.close()

  #where is the subarry data in the full frame?
  subarr=np.where(wht_data != 0)
  xmin,ymin = min(subarr[1]),min(subarr[0])
  xmax,ymax = max(subarr[1]),max(subarr[0])

  #scale the image for display
  interval = PercentileInterval(99.9)
  interval.get_limits(image_data)
  scl_image = interval(image_data) 
  
  #open the catalog file
  catfile=os.path.join(imdrizzlepath,filter+'_prep.cat')
  cat = ascii.read(os.path.join(imdrizzlepath,catfile))

  #get the source IDs, x-positions, and y-positions
  src_id = cat['NUMBER']
  src_x = cat['X_IMAGE']
  src_y = cat['Y_IMAGE']
  
  #create a figure
  fig = plt.figure(1)

  #add a subplot in the bottom left
  sub = fig.add_subplot(1,1,1)

  #low=np.where(scl_image < 0.7)
  #scl_image[low]=0.
  
  #populate the subplot with our image
  sub.imshow(scl_image**2,cmap='gray_r')
  print(scl_image.min(),scl_image.max())
  
  #limit the axes to the subarray data
  sub.set_ylim([ymin,ymax])
  sub.set_xlim([xmin,xmax])

  #circle the catalog sources on the plot
  sub.scatter(src_x,src_y,s=130,marker='o',edgecolors='blue',facecolors='none',lw=1.5)
 
  #Identify source by ID on plot
  for i,junk in enumerate(src_id):
    print(src_id[i],src_x[i],src_y[i])
    sub.annotate('%d'%src_id[i],xy=(src_x[i],src_y[i]),xytext=(src_x[i]+5.,src_y[i]+5.), color='blue',fontsize=14)

  #Save plot
  plt.title(filter)
  plt.savefig(os.path.join(imdrizzlepath,filter+".pdf"))
  
  #Show plot
  plt.show()

  #Ask user to identify target
  target = float(raw_input("Identify the target by entering a number:"))
  
  #Create a new catalog file only containing the target for the aXe run:
  cat.meta={} #scrub the empty lines of header
  discard = [i for i, junk in enumerate(src_id) if src_id[i] != target]
  pdb.set_trace()
  cat.remove_rows(discard)
  ascii.write(cat, os.path.join(imdrizzlepath,filter+'_prep_target.cat'))

def do_axe_prepare(filters,imdrizzlepath):
  working_dir = os.getcwd()
  if target_name == None:
    target_name = 'Target'
    
  #Get an object list for every direct image (Input Object Lists or IOL)
  #We will run this task from within the imdrizzlepath since req'd files are here
  os.chdir(imdrizzlepath)
   
  for filter in filters:
    if filter[0].lower() == 'f': #this is an image, create the IOLs
      catname = filter+'_prep.cat'
      iraf.unlearn('iolprep') #unlearn sets defaults
      iraf.iolprep(filter+'_drz.fits',catname,dimension_info='100,0,0,0',useMdriz='no')
    working_dir = os.path.join(prog_dir,os.environ["AXE_BASE_PATH"])
    #pdb.set_trace()
    os.chdir(prog_dir)
    all_files = glob.glob(os.path.join(os.environ["AXE_FULLFRAME_PATH"],"*flt.fits"))
    i = 0
    while i < len(all_files):
        fits = pyfits.open(all_files[i])
        filter = fits[0].header['FILTER']
        fits.close()
        if filter[0].lower() != 'f':
            all_files.remove(all_files[i])
        else:
            i = i + 1
    for filter in filters:
        if filter[0].lower() == 'g': #grism -- build the lists for aXe
            f = open(os.path.join(os.environ["AXE_FULLFRAME_PATH"],filter+".lis"),'w')
            for imgfile in filters[filter]:
                (imgpath,imgname) = os.path.split(imgfile)
                fits = pyfits.open(imgfile)
                obstime = datetime.datetime.strptime(fits[0].header["DATE-OBS"]+'T'+fits[0].header["TIME-OBS"],"%Y-%m-%dT%H:%M:%S")
                fits.close()
                reference = None
                overall_delta = abs(obstime - datetime.datetime.strptime("1000-01-01T00:00:00","%Y-%m-%dT%H:%M:%S"))
                for reffile in all_files:
                    (refpath,refname) = os.path.split(reffile)
                    fits = pyfits.open(reffile)
                    reftime = datetime.datetime.strptime(fits[0].header["DATE-OBS"]+'T'+fits[0].header["TIME-OBS"],"%Y-%m-%dT%H:%M:%S")
                    fits.close()
                    if reftime < obstime:
                        if abs(obstime-reftime) < overall_delta:
                            overall_delta = abs(obstime-reftime)
                            reference = reffile
                if reference is None:
                    for reffile in all_files:
                        (refpath,refname) = os.path.split(reffile)
                        fits = pyfits.open(reffile)
                        reftime = datetime.datetime.strptime(fits[0].header["DATE-OBS"]+'T'+fits[0].header["TIME-OBS"],"%Y-%m-%dT%H:%M:%S")
                        fits.close()
                        if abs(obstime-reftime) < overall_delta:
                            overall_delta = abs(obstime-reftime)
                            reference = reffile
                (refpath,refname) = os.path.split(reffile)
                f.write("%s %s %s\n" % (imgname,refname.replace("_flt.fits","_flt_1.cat"),refname))
            f.close()
            shutil.copy(os.path.join(os.environ["AXE_FULLFRAME_PATH"],filter+".lis"),os.path.join(os.environ["AXE_BASE_PATH"],filter+".lis"))
            files = glob.glob(os.path.join(os.environ["AXE_FULLFRAME_PATH"],"*.fits"))
            for file in files:
                (p,fname) = os.path.split(file)
                out_file = os.path.join(os.environ["AXE_IMAGE_PATH"],fname)
                shutil.copy(file,out_file)
            files = glob.glob(os.path.join(os.environ["AXE_IMDRIZZLE_PATH"],"i*.cat"))
            for file in files:
                (p,fname) = os.path.split(file)
                out_file = os.path.join(os.environ["AXE_IMAGE_PATH"],fname)
                shutil.copy(file,out_file)
            iraf.unlearn('axeprep')
            #pdb.set_trace()
            iraf.axeprep(os.path.join(os.environ["AXE_BASE_PATH"],filter+".lis"),"WFC3.IR."+filter+".V2.5.conf",backgr='yes',backims="WFC3.IR."+filter+".sky.V1.0.fits",norm='no',mfwhm=10)
    os.chdir(old_dir)
