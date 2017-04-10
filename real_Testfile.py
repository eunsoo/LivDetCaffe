# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 09:12:48 2016

@author: park
"""
#%% This is a real Test code
"""
    Usage : 
        python real_Testfile.py -f /home/park/data/LIVDETECT/Model64 -y LivDet2015 -d conv3_deploy.prototxt \
        -w caffebi/conv3_iter_30000.caffemodel -s Hi_Scan \
        -r /home/park/data/LIVDETECT/conv3_64/ -p 64
"""
import sys
import caffe
import os
import util_livdet as ut
import TestUtil as tu
#from collections import Counter
import numpy as np
import pickle
#%%
caffe_root = '/home/park/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')


def ListToPickle(listfile, destFolder, fileName, force=False):
    fName = os.path.join(destFolder, fileName+'.pickle')
    if (os.path.isfile(fName)) and (not force):
        print "The same name file is exist"
    else:
        f = open(fName, 'w')
        pickle.dump(listfile, f)
        f.close()

    
    

#%% Model Loading 
#model_home = '/home/park/data/LIVDETECT/Model'
#years = ['LivDet2011', 'LivDet2013', 'LivDet2015']
#year = years[0]
#%% So you can input option parameters
from optparse import OptionParser

use = "Usage : %prog [option]"
parser = OptionParser(usage=use)
parser.add_option("-d","--deploy", dest= "define",
                  default="deploy.prototxt", help="deploy file name")
parser.add_option("-w","--weight", dest='weight',
                  default="caffebi/vgg_iter_100000.caffemodel", help="caffemodel name")
parser.add_option("-f", "--fold", dest="m_home",
                  default="/home/park/data/LIVDETECT/Model", help="model home folder")  
parser.add_option("-y", "--year", dest="year",
                  default="LivDet2013", help="DB selection")
parser.add_option("-s", "--sensor", dest="sensor",
                  default="BiometrikaTrain", help="Sensor name (folder)")
parser.add_option("-r", "--result", dest="resultFolder", 
                  default="/home/park/data/LIVDETECT/TestResult", help="result folder name")
parser.add_option("-t", "--testData", dest="testData",
                 default="/home/park/mnt/DBs/FAKE/lvedet/Old_Datasets", help="home of test image")
parser.add_option("-p", "--pSize", dest="patchSize",
                 default=32, help="Size of Patch")


options, args = parser.parse_args()

#%%
model_home = options.m_home
model_def = os.path.join(model_home, options.year, options.sensor, options.define)
model_weight = os.path.join(model_home, options.year, options.sensor, options.weight)
year = options.year
sensor = options.sensor
patchSize = int(options.patchSize)
#testDB = options.testData
#%% Checking path
assert os.path.exists(model_def)
assert os.path.exists(model_weight)
#assert os.path.exists(model_solver)
#%% If you want to run all sensor, you can use this function
#=============================================================
#sensors = ut.loadSensorList(year)
#=============================================================

#%% Caffe loading
#==============================================================
caffe.set_device(0)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()
net = caffe.Net(model_def, model_weight, caffe.TEST)
#==============================================================

#%% Image Transformer
#==============================================================
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#==============================================================

#%% Make Result folders
#==============================================================
testFolder = os.path.join(options.resultFolder, year, sensor)

if year == "LivDet2011":
    label = ut.load2011labels(sensor)
elif year == "LivDet2013":
    label = ut.load2013labels(sensor)
elif year == "LivDet2015":
    label = ut.load2015labels(sensor)
else:
    print "We can not find the Sensor Folder."
    try :    
        raise NameError("NameError")
    except NameError:
        print("Destination root folder name is wrong")
        raise

#%% make Desitination Directories
#=====================================================    
# key value extractino from label table
#tfolders = label.keys()
tfolders = label.keys()
tfolders.remove('bg')
#%%
for tfold in tfolders:
    makingF = os.path.join(testFolder, tfold)
    if os.path.isdir(makingF):
        print makingF + " is already exist."
    else:
        os.makedirs(makingF)
        print "Make Folders : " + makingF
#=====================================================

#%% === Test image loading
#=====================================================
if year == 'LivDet2015':
    fake = 'Fake'
else:
    fake = "Spoof"
#%%
testSensor = sensor.replace("Train","Test")
testDB = os.path.join(options.testData, year, 'Testing', testSensor)
#%%
for tfold in tfolders:
    if tfold is not 'Live':
        testDBFolder = os.path.join(testDB, fake, tfold)
    else:
        testDBFolder = os.path.join(testDB, tfold)

    fileList = ut.search(testDBFolder, ['.png', '.bmp'])
    saveResultFolder = os.path.join(testFolder, tfold)
    for imgName in fileList:
        img = caffe.io.load_image(imgName)
        patch, patchCoor = tu.ExtractPatch(img, patchSize)
        patchNum = len(patch)
        net.blobs['data'].reshape(patchNum,3, patchSize, patchSize)
        for i, pat in enumerate(patch):
            transformed_img = transformer.preprocess('data', pat)
            net.blobs['data'].data[i,...] = transformed_img
        out = net.forward()
        fc12 = net.blobs['fc12'].data
        refc12 = fc12.argmax(axis=1)
        result = []
        for i, ll in enumerate(refc12):
            result.append((patchCoor[i],  ll, fc12[i, ll]))
        ListToPickle(result, saveResultFolder, imgName.split('/')[-1][:-4])
    print(tfold+' is finished')
#        ListToTxtSave(saveResultFolder, imgName.split('.')[0]+'.txt')
#        
#        saveName = os.path.join(saveResultFolder, imgName.split('.')[0])
#        np.save(saveName, result)
        