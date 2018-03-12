import os
import sys
import glob
import shutil

import numpy as np
import matplotlib.pyplot as plt

import pyfits
from pyraf import iraf
from iraf import stsdas, analysis, slitless, axe
from stsci.tools import teal
from astropy.io import fits, ascii
from astropy.visualization import PercentileInterval
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

def replace_mag_auto_with_plam(plam,catalog,new_catalog_name):
  '''
  Change the column heading in a SExtractor catalog from MAG_AUTO to
  MAG<filter pivot wavlength> to prepare catalog for axeprep
  This is instructed on pg 13 of the WFC3 IR grism handbook by J. Lee.

  INPUTS:
  _______
 plam : float
    pivot wavelength ob observations

  catalog : string
    path to catalog file

  new_catalog_name : string
    filename of new catalog to be created

  '''
  cat = open(catalog,'r')
  outpath, fname = os.path.split(catalog)
  outfile = os.path.join(outpath,new_catalog_name)
  out_cat = open(outfile,'w')
  for line in cat.readlines():
    if 'MAG_AUTO' in line:
      line = line.replace('MAG_AUTO','MAG_F%g'%(plam))
    out_cat.write(line)
  out_cat.close()
  cat.close()

def setup_environment(path):
  dirs = {}
  dirs['basepath'] = [path ,  "AXE_BASE_PATH"]
  dirs['savepath'] = [os.path.join(path,"save") ,  "AXE_SAVE_PATH"]
  dirs['datapath'] = [os.path.join(path,"DATA"), "AXE_IMAGE_PATH"]
  dirs['fullframepath'] = [os.path.join(path,"FULLFRAME"), "AXE_FULLFRAME_PATH"]
  dirs['imdrizzlepath'] = [os.path.join(path,"IMDRIZZLE"), "AXE_IMDRIZZLE_PATH"]
  dirs['drizzlepath'] = [os.path.join(path,"DRIZZLE"), "AXE_DRIZZLE_PATH"]
  dirs['outputpath'] = [os.path.join(path,"OUTPUT"), "AXE_OUTPUT_PATH"]
  dirs['confpath'] = [os.path.join(path,"CONF"), "AXE_CONFIG_PATH"]
  
  for item in dirs:
    if not os.path.exists(dirs[item][0]) :  os.makedirs(dirs[item][0])
    os.environ[dirs[item][1]] = dirs[item][0]
  return(dirs)

def copy_data(datalocation,savepath):
  files = glob.glob(os.path.join(datalocation,'*q_flt.fits')) + glob.glob(os.path.join(datalocation,'*q_spt.fits'))
  for file in files:
    try:
      shutil.copy(file,os.path.join(savepath,os.path.split(file)[1]))
    except OSError:
      print('Unable to copy %s from datalocation to save/'%(file))
      
def parse_fits_header(fits_file, keyword):
  hdul = fits.open(fits_file)
  return(hdul[0].header[keyword])
          
def embed_subarrays(savepath,fullframepath):
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
  olddir = os.getcwd()
  
  #If the fullframe path doesn't exist, then create it
  if not os.path.exists(fullframepath):
    os.makedirs(fullframepath)
    
  #Get a list of all q _flt.fits files in the savepath
  files = glob.glob(os.path.join(savepath,'*q_flt.fits'))
  
  #Copy original _flt and _spt files to fullframe directory to avoid modification
  #These files are needed for the embed_subs script
  # (But only do this if the fullframe files don't already exist)
  for file in files:
    fullframe_filename=file.replace("q_flt","f_flt")
    spt_file=file.replace("_flt","_spt")
    if not os.path.exists(os.path.join(fullframepath,fullframe_filename)):
      shutil.copy(file,os.path.join(fullframepath,os.path.split(file)[1]))
      shutil.copy(spt_file,os.path.join(fullframepath,os.path.split(spt_file)[1]))

  #Create full frame images for all *q_flt.fits files in fullframe directory
  #Do this from the fullframepath to avoid long filesnames in fits headers
  #(This will cause problems later on, as files name strings get cut off...)
  os.chdir(fullframepath)
  subfiles = glob.glob('*q_flt.fits')
  embed_subs(subfiles)        
  os.chdir(olddir)
  
  #Now go ahead and delete the copies of the original _flt and _spt files
  copies = glob.glob(os.path.join(fullframepath,"*q_flt.fits")) + glob.glob(os.path.join(fullframepath,"*spt.fits"))
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
      if not os.path.isfile(os.path.join(imdrizzlepath,filter+'_drz.fits')):
        old_dir = os.getcwd()
        os.chdir(imdrizzlepath)  #do the stacking in the imdrizzle directory
        astrodrizzle.AstroDrizzle('@%s.lis'%(filter),output=filter,build='yes',skysub=False)
        os.chdir(old_dir)  #switch back to original working directory


def do_sextractor(filters,imdrizzlepath,confpath):
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
      catalog = os.path.join(imdrizzlepath,filter+'.cat')
      if not os.path.isfile(catalog):

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
        #We are running imcopy in the imdrizzlepath to avoid long pathnames in output .fits headers
        #(If we don't do this header info gets cut off, causing problems later)
        os.chdir(imdrizzlepath)
        iraf.imcopy('%s_drz.fits[SCI]'%(filter),'%s_drz_sci.fits'%(filter))
        iraf.imcopy('%s_drz.fits[WHT]'%(filter),'%s_drz_wht.fits'%(filter)) 
        os.chdir(working_dir)
        
        #Look for SExtractor default files in working directory, and copy to imdrizzlepath
        default_files=['default.sex','default.param','default.conv']
        missing_default_files = [df for df in default_files if not os.path.isfile(os.path.join(imdrizzlepath,df))];
        if len(missing_default_files) > 0:
          for df in missing_default_files:
            try:
              shutil.copy(os.path.join(confpath,df),os.path.join(imdrizzlepath,df))
            except IOError, e:
              print "Unable to copy file. %s" % e

        #Run SExtractor in imdrizzlepath, then change back to working_dir
        #SExtractor creates the .cat file with source positions
        os.chdir(imdrizzlepath)
        zp = wfc3_zeropoint_mags()[filter]
        os.system('sex -c default.sex -WEIGHT_IMAGE %s_drz_wht.fits -MAG_ZEROPOINT %f -CATALOG_NAME %s.cat %s_drz_sci.fits' % (filter,zp,filter,filter)) 
        os.chdir(working_dir)

        new_catalog=os.path.join(imdrizzlepath,filter+'_prep.cat')
        try:
          os.remove(new_catalog)
        except OSError:
          pass
        plam=wfc3_photplam()[filter]
        replace_mag_auto_with_plam(plam,catalog,new_catalog)
        
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
  
  #populate the subplot with our image
  sub.imshow(scl_image**2,cmap='gray_r')
  
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
  cat.remove_rows(discard)
  ascii.write(cat, os.path.join(imdrizzlepath,filter+'_prep_target.cat'))

def do_axeprep(filters,fullframepath,imdrizzlepath,basepath,datapath):
  olddir = os.getcwd()
    
  #Get an object list for every direct image (Input Object Lists or IOL)
  #We will run this task from within the imdrizzlepath since req'd files are here
  #The task iolprep generates an individual object catalogue for each of the direct images
  os.chdir(imdrizzlepath)
  for filter in filters:
    if filter[0].lower() == 'f': #this is an image, create the IOLs
      catalog = filter+'_prep.cat'
      iraf.unlearn('iolprep') #unlearn sets defaults
      iraf.iolprep(filter+'_drz.fits',catalog,dimension_info='+100,0,0,0', useMdriz='no')

  #make a list of all image files
  all_files = glob.glob(os.path.join(fullframepath,"*flt.fits"))
  image_files = [file for file in all_files if parse_fits_header(file, 'FILTER').lower()[0] == 'f']
  
  #get the MJD for each image
  mjd_image=[]
  for file in image_files:
    mjd_image.append(parse_fits_header(file,'EXPSTART'))
  mjd_array=np.array(mjd_image)
  
  #Associate an image file with each grism exposure
  #Writes <grism file> <im cat file> <im file> to .lis file
  for filter in filters:
    if filter[0].lower() == 'g': #grism, continue
      #open list file to write to
      listfile = os.path.join(fullframepath,filter+".lis")
      f = open(listfile,'w')
      #get the MJD for each grism image
      grism_files = filters[filter]
      for file in grism_files:
        gpath, gfile = os.path.split(file)
        mjd = float(parse_fits_header(file,'EXPSTART'))
        #Want the image that is closest in time before the grism exposure
        time_diff = mjd - mjd_array
        if True in (time_diff > 0):  #grism has image taken before it
          ind = np.where(time_diff[time_diff > 0].min() == time_diff)
          associated_image = image_files[ind[0]] 
        else:
          ind = np.where(time_diff.max() == time_diff)
          associated_image = image_files[ind[0]]
        ipath, ifile = os.path.split(associated_image) 
        f.write("%s %s %s\n" % (gfile,ifile.replace("_flt.fits","_flt_1.cat"),ifile))
      f.close()        

      #copy file list into the base directory
      shutil.copy(listfile,os.path.join(basepath,os.path.split(listfile)[1]))

  #copy all _flt.fits files from FULLFRAME/ to DATA/
  flt_files = glob.glob(os.path.join(fullframepath,'*_flt.fits'))
  for file in flt_files:
    shutil.copy(file,os.path.join(datapath,os.path.split(file)[1]))

  #copy all _1.cat file from IMDRIZZLE/ to DATA/
  cat_files = glob.glob(os.path.join(imdrizzlepath,'i*.cat'))
  for file in cat_files:
    shutil.copy(file,os.path.join(datapath,os.path.split(file)[1]))

  #Run axprep: pre-processing of grism images (subtraction of a master sky)
  #This is from from the basepath (will rely on preset environmental variables)
  os.chdir(basepath)
  for filter in filters:
    if filter[0].lower() == 'g': #grism, continue
      iraf.unlearn('axeprep')
      listfile=filter+".lis"
      conffile='WFC3.IR.'+filter+'.V2.5.conf'
      background_image_files = "WFC3.IR."+filter+".sky.V1.0.fits"
      iraf.axeprep(listfile,conffile,backgr='yes',backims=background_image_files,norm='no',mfwhm=7)
  os.chdir(olddir)
  
def do_axecore(filters,basepath,back='yes',extrfwhm=4.0,drzfwhm=0.0,backfwhm=10.0,orient='no',slitless_geom='no',cont_model='gauss',sampling='drizzle',np=5,interp=1):
  olddir = os.getcwd()
  os.chdir(basepath)
  for filter in filters:
    if filter[0].lower() == 'g': #grism, continue
      listfile = os.path.join(basepath,filter+'.lis')
      conffile = "WFC3.IR."+filter+".V2.5.conf"
      iraf.unlearn('axecore')
      iraf.axecore(listfile,conffile,extrfwhm=extrfwhm,drzfwhm=drzfwhm,back=back,backfwhm=backfwhm,orient=orient,slitless_geom=slitless_geom,cont_model=cont_model,sampling=sampling,np=np, interp=interp)
  os.chdir(olddir)   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p","--path",action="store",dest="path",default=os.getcwd(),metavar="PATH",help="Path for data reduction")
    parser.add_argument("-i","--imdrizzle",action="store",dest="imdrizzle_path",default="IMDRIZZLE",help="Path to imdrizzle folder")
    parser.add_argument("-b","--background",action="store_false",dest="background",default=True,help="Switch background subtraction in axecore")
    results = parser.parse_args()

    setup_directories(results.path)
    embed_subarrays()
    filters = filter_lists()
    do_astrodrizzle(filters)
    do_sextractor(filters)
    do_catalogue(filters)
    do_axe_prepare(filters)
    do_axe_run(filters,background=results.background)

