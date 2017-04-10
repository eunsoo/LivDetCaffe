# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:37:46 2016

@author: park
"""
#%%
import numpy as np
import os
import util_livdet as ut
from PIL import Image
from optparse import OptionParser

#%%
def makeLiveAugmentation(sourcePath, resultPath, year, sensor, resultF=None, onlyFake=False):
    sourceFile = open(sourcePath, 'r')
    lines = sourceFile.readlines()
    sourceFile.close()
    #resultFile = open(resultPath, 'w')
    resultFile = []
    sLabel = ut.loadLabelFromYear(year, sensor)
    flip = True
    rotLeft =True
    rotRight = True
    topBottom = True
#    resultF = os.getcwd()
    for line in lines:
        splitted = line.strip().split(' ')
        imageName = os.path.basename(splitted[0])
        label = int(splitted[1])
    #    print imageName, label
        if label == sLabel['Live']:
            resultFile.append(line)
            im = Image.open(splitted[0])
            iName = imageName.split('.')[0]
            ext = '.'+imageName.split('.')[1]
            if resultF is not None:
                dirname = resultF
            else:
                dirname = os.path.dirname(splitted[0])

            if flip is True:
                lrName = iName+'_Aug_lr'+ext
                totalName = os.path.join(dirname, lrName)
                if os.path.isfile(totalName) is not True:
                    lr= im.transpose(Image.FLIP_LEFT_RIGHT)
                    lr.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
                
            if rotLeft is True:
                r90Name = iName+'_Aug_r90'+ext
                totalName = os.path.join(dirname, r90Name)
                if os.path.isfile(totalName) is not True:
                    r90= im.transpose(Image.ROTATE_90)
                    r90.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
                
            if rotRight is True:
                r270Name = iName+'_Aug_r270'+ext
                totalName = os.path.join(dirname, r270Name)
                if os.path.isfile(totalName) is not True:
                    r270 = im.transpose(Image.ROTATE_270)
                    r270.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
                
            if topBottom is True:
                topName = iName+'_Aug_tb'+ext
                totalName = os.path.join(dirname, topName)
                if os.path.isfile(totalName) is not True:
                    tb = im.transpose(Image.FLIP_TOP_BOTTOM)
                    tb.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
        else:
            resultFile.append(line)
            
    
    listToTxtOrg(resultFile ,resultPath)




#%%
def listToTxtOrg(result, fileName):
    f = open(fileName, 'w')    
    for line in result:
        f.write(line)
    f.close()

#%% Augmentation을 함수화 시키자
def makeAugmentation(sourcePath, resultPath, year, sensor, resultF=None, onlyFake=False):
    sourceFile = open(sourcePath, 'r')
    lines = sourceFile.readlines()
    sourceFile.close()
    #resultFile = open(resultPath, 'w')
    resultFile = []
    sLabel = ut.loadLabelFromYear(year, sensor)
    flip = True
    rotLeft =True
    rotRight = True
    topBottom = True
#    resultF = os.getcwd()
    for line in lines:
        splitted = line.strip().split(' ')
        imageName = os.path.basename(splitted[0])
        label = int(splitted[1])
    #    print imageName, label
        if label == sLabel['bg']:
            if onlyFake is False:
                resultFile.append(line)
        elif label is sLabel['Live']:
            if onlyFake is False:
                resultFile.append(line)
        else:
            resultFile.append(line)
            im = Image.open(splitted[0])
            iName = imageName.split('.')[0]
            ext = '.'+imageName.split('.')[1]
            if resultF is not None:
                dirname = resultF
            else:
                dirname = os.path.dirname(splitted[0])
            # Need 4 image transform
            if flip is True:
                lrName = iName+'_Aug_lr'+ext
                totalName = os.path.join(dirname, lrName)
                if os.path.isfile(totalName) is not True:
                    lr= im.transpose(Image.FLIP_LEFT_RIGHT)
                    lr.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
                
            if rotLeft is True:
                r90Name = iName+'_Aug_r90'+ext
                totalName = os.path.join(dirname, r90Name)
                if os.path.isfile(totalName) is not True:
                    r90= im.transpose(Image.ROTATE_90)
                    r90.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
                
            if rotRight is True:
                r270Name = iName+'_Aug_r270'+ext
                totalName = os.path.join(dirname, r270Name)
                if os.path.isfile(totalName) is not True:
                    r270 = im.transpose(Image.ROTATE_270)
                    r270.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
                
            if topBottom is True:
                topName = iName+'_Aug_tb'+ext
                totalName = os.path.join(dirname, topName)
                if os.path.isfile(totalName) is not True:
                    tb = im.transpose(Image.FLIP_TOP_BOTTOM)
                    tb.save(totalName)
                resultFile.append(totalName+' '+str(label)+'\n')
    
    listToTxtOrg(resultFile ,resultPath)

#%%
print "Start Augmentation\n"
use = "Usage : %prog [option]"
parser = OptionParser(usage=use)

parser.add_option("-a","--augtHome", dest='augHome',
                  default="/home/park/data/LIVDETECT/Model", help="Augmentation home")
                  
parser.add_option("-y", "--year", dest="year",
                  default="LivDet2011", help="DB selection")
                  
parser.add_option("-s", "--sensor", dest="sensor",
                  default="BiometrikaTrain", help="Sensor name (folder)")
parser.add_option("-r", "--resultF", dest="resultF",
                  default=None, help="Destination folder name")

options, args = parser.parse_args()

#%%
augHome = options.augHome
year = options.year
sensor = options.sensor
resultF = options.resultF

#%% 이 부분 매우 중요하다. 왜냐면 소스 파일로 부터 aug파일을 만들기 때문이다
# 소스파일의 위치와 같은위치로 aug.txt파일이 저장된다
data_home = 'tr_val'
source_name = 'aug_all_first_train0.txt'
result_name = 'augAugAll_fake_first_train0.txt'




#%%
sourcePath = os.path.join(augHome, year, sensor, data_home, source_name)
resultPath = os.path.join(augHome, year, sensor, data_home, result_name)

# onlyFake 가 트루일경우엔 Live랑 bg 클래스는 제외 된다
#makeAugmentation(sourcePath, resultPath, year, sensor, onlyFake=True)

makeLiveAugmentation(sourcePath, resultPath, year, sensor)
