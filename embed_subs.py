# embed_subs.py: Can be used to embed WFC3 images taken in subarray
# readout mode into a blank full-frame image. Requires the _flt.fits
# and _spt.fits files as input for each image to be processed.
# Wildcards may be used in the input string, e.g. *flt.fits, in order
# to process many files in one run.
#
# A new output _flt.fits file will be created for each input file,
# with the new file having the same rootname as the old, but with the
# final character (often a "q") replaced with the letter "f". The full
# SCI, ERR, DQ structure (for UVIS exposures) and SCI, ERR, DQ, SAMP, TIME
# structure (for IR exposures) is reproduced in the output _flt file.
# The SCI and ERR values for the pixels outside the original subarray
# region will be set to zero, and the DQ values will be set to 4, in
# order to indicate that those pixels should not be used in any subsequent
# processing or analysis.
#
# H. Bushouse (STScI)
# Sept. 2011
#
import pyfits
import glob
import numpy

def embed_subs(flist):
    # Set some image dimension constants to be used later
    uvis_a1_size = 4096
    uvis_a2_size = 2051
    serial_over = 25.0
    ir_a1_size = 1014
    ir_a2_size = 1014
    ir_overscan = 5.0
    
    if len(flist) == 0:
        print ("ERROR: No files found JR")
    
    # Loop over the list of input images
    for file in flist:
    
        # Make sure the input name conforms to normal style
        if file.find('_flt') < 0:
            print ("Warning: Can't properly parse '%s'; Skipping" % file)
            continue
    
        # Extract the root name and build SPT and output file names
        root = file[0:file.find('_flt')]
        flt = file
        spt = root + '_spt.fits'
        full = root[0:len(root)-1] + 'f_flt.fits'
    
        print (" Processing input files '%s' and '%s' ..." % (flt,spt))
    
        # Open the input FLT and SPT files
        try:
            fd1=pyfits.open(flt, 'readonly')
        except IOError:
            print ("ERROR: Can't open file %s" % flt)
            continue
        try:
            fd2=pyfits.open(spt, 'readonly')
        except IOError:
            print ("ERROR: Can't open file %s" % spt)
            fd1.close()
            continue
    
        # Get keyword values that describe image type and size
        detector=fd1[0].header['DETECTOR']
        subarray=fd1[0].header['SUBARRAY']
        xcorner=int(fd2[1].header['XCORNER'])
        ycorner=int(fd2[1].header['YCORNER'])
        numrows=int(fd2[1].header['NUMROWS'])
        numcols=int(fd2[1].header['NUMCOLS'])
        sizaxis1 = numcols
        sizaxis2 = numrows
    
        # Get crpix values that will need modifying later
        crpix1 = fd1[1].header['CRPIX1']
        crpix2 = fd1[1].header['CRPIX2']
        if 'ocrpix1' in fd1[1].header:
            ocrpix1 = fd1[1].header['ocrpix1']
            ocrpix2 = fd1[1].header['ocrpix2']
        elif 'crpix1o' in fd1[1].header:
            ocrpix1 = fd1[1].header['crpix1o']
            ocrpix2 = fd1[1].header['crpix2o']
    
        # Skip this image if it's not a subarray
        if (not subarray):
            print ("Warning: File '%s' is not a subarray image; Skipping." % flt)
            fd1.close()
            fd2.close()
            continue
    
        # Compute UVIS subarray corners
        if detector == 'UVIS':
            cornera1 = ycorner
            cornera2 = uvis_a2_size - xcorner - sizaxis2
            if xcorner >= uvis_a2_size:
                cornera2 = cornera2 + uvis_a2_size
            cornera1a = cornera1 + 1 - serial_over
            cornera1b = cornera1a + sizaxis1 - 1
            cornera2a = cornera2 + 1
            cornera2b = cornera2a + sizaxis2 - 1
            if cornera1a < 1:
                cornera1a = 1
            if cornera1b > uvis_a1_size:
                cornera1b = uvis_a1_size
    
            # Set full-chip axis sizes
            fnaxis1 = uvis_a1_size
            fnaxis2 = uvis_a2_size
    
        # Compute IR subarray corners
        else:
            cornera1 = ycorner - ir_overscan
            cornera2 = xcorner - ir_overscan
            cornera1a = cornera1 + 1
            cornera1b = cornera1a + sizaxis1 - 11
            cornera2a = cornera2 + 1
            cornera2b = cornera2a + sizaxis2 - 11
    
            # Set full-chip axis sizes
            fnaxis1 = ir_a1_size
            fnaxis2 = ir_a2_size
    
    
        # Report computed subarray section
        print (" Subarray image section = [%d:%d,%d:%d]" % (cornera1a,cornera1b,cornera2a,cornera2b))
    
        # Create output full-chip file
        print (" Creating output file '%s' ..." % full)
    
        # First just copy the primary header to the output
        try:
            pyfits.writeto(full, fd1[0].data, fd1[0].header)
        except IOError:
            print ("ERROR: Can't create/write file '%s'" % full)
            fd1.close()
            fd2.close()
            continue
    
        # Now copy the subarray image data into full-chip data arrays;
        # The regions outside the subarray will be set to zero in the
        # SCI, ERR, SAMP, and TIME extensions, and to DQ=4.

        #JR modificaiton: set the region outside the subarry to the MEDIAN of the subarray to allow for sky subtraction, note this is commented out because it doesn't actually matter with DQ set to 4 in this area.
        
       # subarray_median = numpy.median(fd1[1].data)
        sci = numpy.zeros([fnaxis2,fnaxis1],dtype=numpy.float32)# + subarray_median
        err = numpy.zeros([fnaxis2,fnaxis1],dtype=numpy.float32)
        dq  = numpy.zeros([fnaxis2,fnaxis1],dtype=numpy.int16) + 4
        sci[cornera2a-1:cornera2b,cornera1a-1:cornera1b] = fd1[1].data
        err[cornera2a-1:cornera2b,cornera1a-1:cornera1b] = fd1[2].data
        dq[cornera2a-1:cornera2b,cornera1a-1:cornera1b]  = fd1[3].data
        if detector == 'IR':
            samp = numpy.zeros([fnaxis2,fnaxis1],dtype=numpy.int16)
            time = numpy.zeros([fnaxis2,fnaxis1],dtype=numpy.float32)
            samp[cornera2a-1:cornera2b,cornera1a-1:cornera1b] = fd1[4].data
            time[cornera2a-1:cornera2b,cornera1a-1:cornera1b] = fd1[5].data
    
        # Reset a few WCS values to make them appropriate for a
        # full-chip image
        fd1[1].header['sizaxis1'] = fnaxis1
        fd1[1].header['sizaxis2'] = fnaxis2
        for i in range(1,4):
            fd1[i].header['crpix1'] = crpix1 + cornera1a - 1
            fd1[i].header['crpix2'] = crpix2 + cornera2a - 1
            fd1[i].header['ltv1']   = 0.0
            fd1[i].header['ltv2']   = 0.0
            if 'onaxis1' in fd1[i].header:
                fd1[i].header['onaxis1'] = fnaxis1
            if 'onaxis2' in fd1[i].header:
                fd1[i].header['onaxis2'] = fnaxis2
            if 'ocrpix1' in fd1[i].header:
                fd1[i].header['ocrpix1'] = ocrpix1 + cornera1a - 1
            if 'ocrpix2' in fd1[i].header:
                fd1[i].header['ocrpix2'] = ocrpix2 + cornera2a - 1
    
        # Now write out the SCI, ERR, DQ extensions to the full-chip file
        pyfits.append(full, sci, fd1[1].header)
        pyfits.append(full, err, fd1[2].header)
        pyfits.append(full, dq,  fd1[3].header)
        if detector == 'IR':
            pyfits.append(full, samp, fd1[4].header)
            pyfits.append(full, time, fd1[5].header)
    
        # flush the output file
        fd3 = pyfits.open(full,'update')
        fd3[0].header["SUBARRAY"] = False #needed because of check later in aXe for subarray data.
        fd3.close()
    
        # close the input files
        fd1.close()
        fd2.close()

if __name__ == "__main__":
    # Get the list of input FLT images from the user
    pattern = raw_input ("Enter FLT file name(s): ")
    flist = glob.glob(pattern)

    embed_subs(flist)
