# -*- coding: utf-8 -*-
import segmentation as seg
import util_livdet as util
import os
import cv2
from optparse import OptionParser
# Main function
if __name__ == "__main__":

######### Patch Maker #####################################
#==============================================================================
# """
# This code makes patch data that has the same original folder architecture.
# It return patches and the name of patch images in txt format. This txt file can be
# used in Caffe layer for traning.
# 
# Uage : python makePatch.py -y LivDet2011 -s 16  
# """
	use = "Usage : %prog [option]"
	parser = OptionParser(usage=use)

	parser.add_option("-y","--year", dest='year',
                  default="LivDet2011", help="Year of Sensors")
	parser.add_option("-s","--step", dest='step',
                  default=16, help="Step Size")
	options, args = parser.parse_args()

	# You should keep the folder ordering of Livdet dataset
	my_livdet = "/home/park/mnt/DBs/FAKE/lvedet/Old_Datasets" # It should be your daset folder 
	dataset = options.year
	rootDir = "/home/park/mnt/DBs/FAKE/lvedet/Old_Datasets/"+dataset
	trFolderName = "Training"
	patchName = "Patch"
	stepSize = int(options.step)
	rate = 0.70
	bgName = 'bg'
	fgName = 'fg'
	orFolderName = os.path.join(rootDir, trFolderName)
	patchFolderName = os.path.join(rootDir, patchName, str(stepSize)+'x'+str(stepSize), trFolderName)

	#%% image and folder search pairs    
	direc, imgFiles = util.folderFiles(orFolderName, ['.bmp','.png'])
	#%% Make Destiantion derectory
	destFolder = util.makeDestFolder(patchFolderName, direc)
	#%% Make patches
	png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
	#%%
	for index, dt in enumerate(destFolder):
		backFolder = os.path.join(dt, bgName)
		foreFolder = os.path.join(dt, fgName)
		orgFolder = direc[index]
		util.makeBGFGFolder(backFolder)        
		util.makeBGFGFolder(foreFolder)        
		
		bgtxtName = util.makeTxtName(dt, bgName, backFolder)
		bfile = open(bgtxtName, 'w')

		fgtxtName = util.makeTxtName(dt, fgName, foreFolder)
		ffile = open(fgtxtName, 'w')        
		
		for ii, img in enumerate(imgFiles[index]):
			imgName = os.path.join(orgFolder, img)
			image = cv2.imread(imgName, 0)
			if image is None: continue
			else:
				segImg = seg.weonjinSegmentation(image)
				bg, fg, bgtl, fgtl = util.makePatchData(image, segImg, stepSize, rate)
				util.patchSave(bg, bgtl, backFolder, png_params, img, bfile)
				util.patchSave(fg, fgtl, foreFolder, png_params, img, ffile)
			
#  patchSave function need to be changed. list of bfile is better. Saving to
#  txt file is latter to finish making list of bfile.
	bfile.close()
	ffile.close()