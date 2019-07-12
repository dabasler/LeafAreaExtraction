#!/usr/bin/env python

"""
# This scripts extracts leaf area of scanned leaves
Created in June 2017
@author: DB
"""

# -*- coding: utf-8 -*-

# Libraries 
import os,sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
#from scipy.ndimage import gaussian_filter
from scipy import ndimage


### FUNCTIONS 

def getRGB(imagename):
	img = Image.open(imagename)
	try:
		dpi = img.info["dpi"][1]
	except:
		dpi=99
	img.load()
	imgR,imgG,imgB = img.split()
	R = np.asarray(imgR)
	G = np.asarray(imgG)
	B = np.asarray(imgB)
	R.flags.writeable = True
	G.flags.writeable = True
	B.flags.writeable = True
	return R,G,B,dpi

def normalize(a):
	a=a*1.0
	m = np.amin(a)
	M = np.amax(a)
	n = 255*(a-m)/(M-m)
	n=np.array(n,dtype=np.uint8)
	return n

def otsu_threshold(image, bins=256):
	"""Return threshold value based on Otsu's method.
	Parameters
	----------
	image : array
		Input image.
	bins : int
		Number of bins used to calculate histogram. This value is ignored for
		integer arrays.
	Returns
	-------
	threshold : numeric
		Threshold value. int or float depending on input image.
	References
	----------
	.. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method
	"""
	hist, bin_centers = histogram(image, bins)
	hist = hist.astype(float)
	# class probabilities for all possible thresholds
	weight1 = np.cumsum(hist)
	weight2 = np.cumsum(hist[::-1])[::-1]
	# class means for all possible thresholds
	mean1 = np.cumsum(hist * bin_centers) / weight1
	mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
	# Clip ends to align class 1 and class 2 variables:
	# The last value of `weight1`/`mean1` should pair with zero values in
	# `weight2`/`mean2`, which do not exist.
	variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
	idx = np.argmax(variance12)
	threshold = bin_centers[:-1][idx]
	return threshold



def binarize(image, method='otsu'):
	"""Return binary image using an automatic thresholding method.
	Parameters
	----------
	image : array
		Input array.
	method : {'otsu'}
		Method used to calculate threshold value. Currently, only Otsu's method
		is implemented.
	Returns
	-------
	out : array
		Thresholded image.
	"""
	get_threshold = _threshold_funcs[method]
	threshold = get_threshold(image)
	return image > threshold


def histogram(image, bins):
	"""Return histogram of image.
	Unlike `numpy.histogram`, this function returns the centers of bins and
	does not rebin integer arrays.
	Parameters
	----------
	image : array
		Input image.
	bins : int
		Number of bins used to calculate histogram. This value is ignored for
		integer arrays.
	Returns
	-------
	hist : array
		The values of the histogram.
	bin_centers : array
		The values at the center of the bins.
	"""
	if np.issubdtype(image.dtype, np.integer):
		if np.min(image) < 0:
			msg = "Images with negative values not allowed"
			raise NotImplementedError(msg)
		hist = np.bincount(image.flat)
		bin_centers = np.arange(len(hist))
		# clip histogram to return only non-zero bins
		idx = np.nonzero(hist)[0][0]
		return hist[idx:], bin_centers[idx:]
	else:
		hist, bin_edges = np.histogram(image, bins=bins)
		bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
		return hist, bin_centers

def threshold_yen(image, nbins=256):
    """Return threshold value based on Yen's method.
    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels intensities that less or equal of
        this value assumed as foreground.
    References
    ----------
    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_yen(image)
    >>> binary = image <= thresh
    """
    hist, bin_centers = histogram(image.ravel(), nbins)
    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns `bin_centers` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]
    # Calculate probability mass function
    pmf = hist.astype(np.float32) / hist.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf ** 2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid '-inf'
    # in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
                  (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]

################################## MAIN FUNCTION ###############################################33
def Extract_LeafArea(filename,output_path,black=False):
	MIN_SIZE=1000
	R,G,B,dpi=getRGB(filename)
	bG=G.copy()
	tv=otsu_threshold(G, bins=256)
	bG[G>tv]=0
	bG[G<=tv]=1
	
	bG[:,0:350]=0 # Remove Reference stripe
	
	s = [[1,1,1],[1,1,1],[1,1,1]]
	bg_labeled,nlabels=ndimage.measurements.label(bG,s)
	# Remove Artefacts < MIN_SIZE
	for i in range(nlabels):
		n=len(np.where(bg_labeled==i)[0])
		if (n<=MIN_SIZE):
			bg_labeled[bg_labeled==i]=0
	# RUN Again to remove some leftover artefacts
	for i in np.unique(bg_labeled):
		n=len(np.where(bg_labeled==i)[0])
		if (n<=MIN_SIZE):
			bg_labeled[bg_labeled==i]=0
	# get rid of any border objects
	for i in np.unique(bg_labeled[0,:]): # TOP
		if np.mean(G[bg_labeled==i])<(np.mean(B[bg_labeled==i])*1.2): # Delete object, but make an exception for very green objects (leafs)
			bg_labeled[bg_labeled==i]=0
	for i in np.unique(bg_labeled[:,0]): # LEFT
		if np.mean(G[bg_labeled==i])<(np.mean(B[bg_labeled==i])*1.2): # Delete object, but make an exception for very green objects (leafs)
			bg_labeled[bg_labeled==i]=0 
	for i in np.unique(bg_labeled[bg_labeled.shape[0]-1,:]): # BOTTOM
		if np.mean(G[bg_labeled==i])<(np.mean(B[bg_labeled==i])*1.2): # Delete object, but make an exception for very green objects (leafs)
			bg_labeled[bg_labeled==i]=0	
	for i in np.unique(bg_labeled[:,bg_labeled.shape[1]-1]): # RIGHT
		if np.mean(G[bg_labeled==i])<(np.mean(B[bg_labeled==i])*1.2): # Delete object, but make an exception for very green objects (leafs)
			bg_labeled[bg_labeled==i]=0
		#print i
		#print len(np.where(bg_labeled==i)[0])
		#print np.mean(G[bg_labeled==i])
		#print np.mean(B[bg_labeled==i])
		#print "**"
		#if np.mean(G[bg_labeled==i])<(np.mean(B[bg_labeled==i])*1.2): # Delete object, but make an exception for very green objects (leafs)
			#bg_labeled[np.logical_and(G[bg_labeled==i]<(B[bg_labeled==i]*1.2),bg_labeled[bg_labeled==i])]=0
			
	# CLEANUP NUBERING
	bgl=bg_labeled.copy()
	j=0
	for i in np.unique(bg_labeled):
		bgl[bg_labeled==i]=j
		j=j+1
	# EXPORT DATA
	leaf_colors=[]
	leaf_area=[]
	n_objects = len(np.unique(bgl))
	
	# Prepare Output File
	fout_ind=open(os.path.join(output_path,os.path.splitext(os.path.split(filename)[-1])[0]+"_ILA.csv"),'w')
	header="file;n;area_cm2;meanR;meanG;meanB;sdR;sdG;sdB\n" 
	fout_ind.write(header)
	if n_objects>1:
		#Individual output
		for i in np.unique(bgl)[1:]:
			print "\tLeaf %i" %i
			n=len(np.where(bgl==i)[0])
			area=((2.54*2.54)/(dpi*dpi))*n
			leaf_area.append(area)
			lR=R[bgl==i]
			lG=G[bgl==i]
			lB=B[bgl==i]
			fout_ind.write("%s;%i;%f;%f;%f;%f;%f;%f;%f\n" %(os.path.split(filename)[-1],i,area,np.mean(lR),np.mean(lG),np.mean(lB),np.std(lR),np.std(lG),np.std(lB)))
			leaf_colors.append([int(np.mean(lR)),int(np.mean(lG)),int(np.mean(lB))])
	else:
			fout_ind.write("%s;NA;NA;NA;NA;NA;NA;NA;NA\n")
			
	fout_ind.close()
	###
	# ReferenceList
	# Reference strip 200x200
	sref=200
	RefStrip=[(61,130),(61,525),(61,976),(61,1397),(61,1773),(61,2218),(61,2643)]
	RefValues=np.zeros(21).reshape(7,3)
	for i in range(0,7):
		p = RefStrip[i]
		RefValues[i,:]=(np.median(R[p[1]:p[1]+sref,p[0]:p[0]+sref]),np.median(G[p[1]:p[1]+sref,p[0]:p[0]+sref]),np.median(B[p[1]:p[1]+sref,p[0]:p[0]+sref]))

	RefMean=np.mean(RefValues, axis=0)
	RefString = "%f;%f;%f" % (RefMean[0],RefMean[1],RefMean[2])

	# LEAF PARAMETERS
	n_pixel = len(np.where(bgl>0)[0])
	
	n_objects = len(np.unique(bgl))-1
	lR=R[bgl>0]
	lG=G[bgl>0]
	lB=[bgl>0]
	
	total_area=((2.54*2.54)/(dpi*dpi))*n_pixel
	mean_area=np.mean(leaf_area)
	sd_area= np.std(leaf_area)
	
	# Prepare Output File
	fout_tot=open(os.path.join(output_path,os.path.splitext(os.path.split(filename)[-1])[0]+"_TLA.csv"),'w')
	refHeader="ref_R;ref_G;ref_B"
	header="file;n;total_area_cm2;mean_areacm2;sd_areacm2;meanR;meanG;meanB;sdR;sdG;sdB;%s\n" % refHeader
	fout_tot.write(header)
	fout_tot.write("%s;%i;%f;%f;%f;%f;%f;%f;%f;%f;%f;%s\n" %(os.path.split(filename)[-1],n_objects,total_area,mean_area,sd_area,np.mean(lR),np.mean(lG),np.mean(lB),np.std(lR),np.std(lG),np.std(lB),RefString))
	fout_tot.close()
	
	#EXPORT REFERENCE IMAGE
	R[:100,:600]=0
	G[:100,:600]=0
	B[:100,:600]=0
	n_objects = len(np.unique(bgl))
	if black==False:	
		R[bgl==0]=R[bgl==0]/4
		G[bgl==0]=G[bgl==0]/4
		B[bgl==0]=B[bgl==0]/4
	else:
		R[bgl==0]=0
		G[bgl==0]=0
		B[bgl==0]=0
	cs=100
	cx=120
	for i in range(n_objects-1):
		cy1=i*(cs+25)+400
		cy2=cy1+cs
		R[cy1:cy2,cx:(cx+cs)]=leaf_colors[i][0]
		G[cy1:cy2,cx:(cx+cs)]=leaf_colors[i][1]
		B[cy1:cy2,cx:(cx+cs)]=leaf_colors[i][2]
	cx=40
	R[3000:3100,cx:(cx+cs)]=RefValues[0,0]
	G[3000:3100,cx:(cx+cs)]=RefValues[0,1]
	B[3000:3100,cx:(cx+cs)]=RefValues[0,2]
	R[3000:3100,(cx+cs):(cx+2*cs)]=RefValues[1,0]
	G[3000:3100,(cx+cs):(cx+2*cs)]=RefValues[1,1]
	B[3000:3100,(cx+cs):(cx+2*cs)]=RefValues[1,2]
	R[3000:3100,(cx+2*cs):(cx+3*cs)]=RefValues[2,0]
	G[3000:3100,(cx+2*cs):(cx+3*cs)]=RefValues[2,1]
	B[3000:3100,(cx+2*cs):(cx+3*cs)]=RefValues[2,2]
	
	R[3100:3200,cx:(cx+cs)]=RefValues[3,0]
	G[3100:3200,cx:(cx+cs)]=RefValues[3,1]
	B[3100:3200,cx:(cx+cs)]=RefValues[3,2]
	R[3100:3200,(cx+cs):(cx+2*cs)]=RefValues[4,0]
	G[3100:3200,(cx+cs):(cx+2*cs)]=RefValues[4,1]
	B[3100:3200,(cx+cs):(cx+2*cs)]=RefValues[4,2]
	R[3100:3200,(cx+2*cs):(cx+3*cs)]=RefValues[5,0]
	G[3100:3200,(cx+2*cs):(cx+3*cs)]=RefValues[5,1]
	B[3100:3200,(cx+2*cs):(cx+3*cs)]=RefValues[5,2]
	
	#mg=G.copy()
	#mg[bgl>0]=240
	#normalize(bgl)
	img=Image.merge("RGB",(Image.fromarray(R),Image.fromarray(G),Image.fromarray(B)))
	s= img.size
	img = img.resize((s[0]/4, s[1]/4), Image.ANTIALIAS)
	draw = ImageDraw.Draw(img)
	font = ImageFont.load("pilfonts/helvR14.pil")
	draw.text((10, 0),os.path.splitext(os.path.split(filename)[-1])[0],(0,0,255),font=font)
	n_objects = len(np.unique(bgl))
	draw.text((10,75),"average color",(255,0,0),font=font)
	draw.text((10,725),"reference",(255,0,0),font=font)
	if n_objects>1:
		#Number individual leafs
		draw.text((330,20),"Mean area: %.2f +- %.2f cm^2" % (mean_area,sd_area),(0,255,0),font=font)
		for i in np.unique(bgl)[1:]:
			ids=np.where(bgl==i)
			n=len(ids[0])
			area=((2.54*2.54)/(dpi*dpi))*n
			x=int(np.min(ids[1])/4)
			y=int(np.max(ids[0])/4)
			draw.text((x,y-20),str(i),(255,0,0),font=font)
			draw.text((x,y),"%.2f cm^2" % area,(255,0,0),font=font)
			draw.text((10,((i-1)*(cs+25)+400)/4),str(i),(255,0,0),font=font)
	
	draw.text((330,0),"Total area: %.2f cm^2" % total_area,(0,255,0),font=font)
	
	img.save(os.path.join(output_path,os.path.splitext(os.path.split(filename)[-1])[0]+"_LA.jpg"), format='JPEG', subsampling=0, quality=95)

#################################################################################33
## Collect individual files and create a summary

def create_summary(filelist,output_path,outfilename,header):
	fout=open(os.path.join(output_path,outfilename),'w')
	fout.write(header)
	for file in filelist:
		f=open(file,'r')
		l=f.readlines()
		x=l.pop(0)
		fout.writelines(l)
	fout.close()

def summarize(output_path):
	print "Writing result summary"
	# Output Individual
	#ind_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f[-7:] == 'ILA.csv' ]
	ind_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(output_path) for f in filenames if f[-7:] == 'ILA.csv']
	header="file;n;area_cm2;meanR;meanG;meanB;sdR;sdG;sdB\n" 
	create_summary(ind_files,output_path,"LeafArea_Individual.csv",header)

	# Output Total
	#tot_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f[-7:] == 'TLA.csv' ]
	tot_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(output_path) for f in filenames if f[-7:] == 'TLA.csv']
	refHeader="ref_R;ref_G;ref_B"
	header="file;n;total_area_cm2;mean_areacm2;sd_areacm2;meanR;meanG;meanB;sdR;sdG;sdB;%s\n" % refHeader
	create_summary(tot_files,output_path,"LeafArea_Total.csv",header)

#################################################################################33
# INITIALIZE AND SETUP

print "\n*** EXTRACT LEAF AREA ***\n"

if len(sys.argv) < 2:
	print ("format: leafArea.py <Inputpath> or leafArea.py <Inputfile>")
	print ("flags:\n\t-F\t force overwrite existing files\n\t-black\t black background\n\t-sum\tproduce summary only\n")
	exit(1)

filename = sys.argv[1]

overwrite=False

if "-F" in sys.argv:
	overwrite=True

black=False
if "-black" in sys.argv:
	black=True


if not os.path.exists(filename):
    print ("ERROR: File/path '%s' does not exist\n\nformat: extract_la.py <Inputpath> or extract_la.py <Inputfile>" % filename)
    exit(1)

if os.path.splitext(filename)[1] == '.jpeg' or os.path.splitext(filename)[1] ==".jpg": # Argument is a FILE
	path=os.path.split(filename)[0]
	if path=='':
		path="."
	filename=[filename]
	
else: # Argument is a PATH
	path=filename
	filename = [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.jpeg']
	outfile=""

output_path=os.path.join(path,"output")
if not os.path.exists(os.path.join(path,"output")):
	os.mkdir(output_path)


if "-sum" in sys.argv:
	summarize(output_path)
	print "Done!"
	sys.exit(0)
	

if "-csum" in sys.argv:
	summarize(path)
	print "Done!"
	sys.exit(0)
	
###########################################
# RUN EXTRACTION

# Loop through files
nfiles=len(filename)
for i in range(nfiles):
	fn=filename[i]
	if overwrite==True:
		print "processing %i/%i: %s" % (i+1,nfiles,fn)
		Extract_LeafArea(fn,output_path,black)
	else:
		if not os.path.exists(os.path.join(output_path,os.path.splitext(os.path.split(fn)[-1])[0]+"_LA.jpg")):
			print "processing %i/%i: %s" % (i+1,nfiles,fn)
			Extract_LeafArea(fn,output_path,black)
		else:
			print "skipping %i/%i: %s \tFILE ALREADY EXISTS" % (i+1,nfiles,fn)
	
summarize(output_path)

print "Done!"
