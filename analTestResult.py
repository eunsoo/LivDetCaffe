# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:50:36 2016

@author: park
"""
#%%
#import pickle
import os
from optparse import OptionParser
import util_livdet as ut
import resultAnalizer as res
import matplotlib.pyplot as plt
import numpy as np

#%%
use = "Usage : %prog [option]"
parser = OptionParser(usage=use)

parser.add_option("-t","--testHome", dest='testHome',
                  default="/home/park/data/LIVDETECT/TestResult", help="test result home")
parser.add_option("-y", "--year", dest="year",
                  default="LivDet2011", help="DB selection")
parser.add_option("-s", "--sensor", dest="sensor",
                  default="BiometrikaTrain", help="Sensor name (folder)")
parser.add_option("-r", "--result", dest="resultF",
                  default="Result", help="Result folder name (folder)")
parser.add_option("-d", "--do", dest="doing",
                  default="pred", help="What do you do")
parser.add_option("-b", "--binary", dest="binary",
                  default=False, help="Binary Class?")

options, args = parser.parse_args()

#%% Initial Value
testHome = options.testHome
year = options.year
sensor = options.sensor
resFolder = options.resultF
doing = options.doing
binary = options.binary

###############################################################################
###############################################################################
###############################################################################
if doing == 'top':
  finalPath = os.path.join(testHome, year, sensor)
      
  label = res.loadLabelFromYear(year, sensor)
  tfolders = label.keys()
  tfolders.remove('bg')
  folderName='LabelScore'
  for tfold in tfolders:
      resultF = os.path.join(finalPath, tfold)
      if os.path.isdir(resultF):
          print(year+', ', tfold, ' is processing')
          fileList = ut.search(resultF, ['.pickle'])
  #        bins = np.empty(1,1)
          for ind in xrange(len(fileList)):
             if ind==0:
                 frame = np.asarray(res.pickleLoader(fileList[ind]))
                 sframe = frame[frame[:,1] == label[tfold]]
                 name = np.asarray(os.path.basename(fileList[ind]).split('.')[0])
                 name = name.repeat(sframe.shape[0]).reshape(sframe.shape[0],1)
                 newSframe = np.concatenate([name, sframe], axis=1)
                 bins = newSframe
             else:
                 frame = np.asarray(res.pickleLoader(fileList[ind]))
                 sframe = frame[frame[:,1] == label[tfold]]
                 name = np.asarray(os.path.basename(fileList[ind]).split('.')[0])
                 name = name.repeat(sframe.shape[0]).reshape(sframe.shape[0],1)
                 newSframe = np.concatenate([name, sframe], axis=1)
                 bins = np.concatenate([bins, newSframe], axis=0)
                 
          resultPath = os.path.join(finalPath, folderName)
          ut.makeBGFGFolder(resultPath)
          # -100 part have to be option of top 
          top100 = bins[bins[:,3].argsort()][-100:]
          fileName = os.path.join(resultPath, tfold)
          res.listToTxt(top100.tolist(), fileName+'.txt', sel='top')
      else:
          print(tfold, ' is not exist.')






###############################################################################
###############################################################################
###############################################################################
###############################################################################

#%% 이걸 실행시키면 피클로 저장되어 있는 이미지별 패치 스코어와 레이블을 모아서
# 실제 보팅 후 결과를 뽑아 내고 그것을 Result 폴더에 저장하게 된다. 이때 그 결과는
# Text 파일로 저장된다.
#=============================================================================
if doing == 'pred':
  print "Doing Prediction to "
  res.patchToDetectionTextResult(testHome, year, sensor, resFolder, binary=binary)
#=============================================================================
#=============================================================================
#=============================================================================




#%%# 이제 이 텍스트 파일들을 또 다시 모아서 confusion matrix를 만들어야 함
# Confusion matrix를 만들고 그림도 그려보는 예제임
#=============================================================================
##=============================================================================
if doing == 'anal':
  refold = os.path.join(testHome, year, sensor, resFolder)
  label = res.loadLabelFromYear(year, sensor)
  # if binary:
  #   label ={'Live':0, 'Fake':1}
  #%% Confusion matrix
  y_test, y_pred = res.makePrediction(refold, label, binary)
  #%% Draw Confusion matrix
  confmat = res.makeConfusionMatrix(y_test, y_pred)
  print "=============== Label Information :"
  if binary :
    label ={'Live':0, 'Fake':1}
  print label
  print "=============== Confusiotn Matrix :"
  print confmat.T
  # res.plotPrecisionAndRecall()
  print "=============== Precision and recall : "
  precision, recall = res.MultiRecallPrecision(confmat)
  print precision
  print recall
 

  # if binary is not True:
  new_t, new_p = res.makeTwoClass(y_test, y_pred)
  new_cfm = res.makeConfusionMatrix(new_t, new_p)
  print "=============== In case of Two Clss"
  print new_cfm.T
  print "=============== Precision and recall : "
  precision, recall = res.MultiRecallPrecision(new_cfm)
  print precision
  print recall
  print "=============== SFPR(Spoof False Positive Rate)"
  sfpr = (1-precision)*100
  print sfpr
  print "=============== SFNR(Spoof False Negative Rate)"
  sfnr = (1-recall)*100
  print sfnr
  print "=============== ACE (Average Classification Error)"
  ace = (sfpr+sfnr)/2
  print ace
#
##=============================================================================

##=============================================================================
##=============================================================================
##=============================================================================
    