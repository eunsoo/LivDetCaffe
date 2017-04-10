# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:40:15 2016

@author: park
"""
"""
  This code is for training your patch based liveness detection module.
  This code uses caffe python api. 
  This code supports continuous training , keeps track of log and draws training test accuracy.

  Usage :
    python pyTrainingRecordLog.py -y LivDet2015 -i Hi_Scan -c True -s caffebi/conv3_iter_30000.solverstate
"""


import os
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
import util_livdet as ut


# python pyTrainingRecordLog.py -c True 
#%%
use = "Usage : %prog [option]"
parser = OptionParser(usage=use)
parser.add_option("-m","--home", dest='model_home',
                  default="/home/park/data/LIVDETECT/Model64", help="Model home")
parser.add_option("-c","--continue", dest='conti',
                  default=False, help="True if you want to continue to train")
parser.add_option("-s","--solstate", dest='solverstate',
                  default="caffebi/vgg2NoVal_iter_80000.solverstate", help="name of solver state")
parser.add_option("-y","--year", dest='year', default="LivDet2011", help="name of year")
parser.add_option("-i","--sensor", dest='sensor', default="BiometrikaTrain", help="name of sensor")
parser.add_option("-v","--solver", dest='solver', default="conv3_solver.prototxt", help="name of solver")
parser.add_option("-l","--log", dest='logfile', default="log/conv3.log", help="name of solver")      
parser.add_option("-a","--all", dest='all', default=False, help="Want all?")      
                  
#%%                  


options, args = parser.parse_args()
allPro = options.all
# allPro = True
#%%
if allPro is True:
  model_home = options.model_home
  years =["LivDet2011","LivDet2013","LivDet2015"]
  solver = options.solver
  logfile= options.logfile
  for year in years:
      sensors = ut.loadSensorList(year)
      for sensor in sensors:
            solverName = os.path.join(model_home, year, sensor, solver)
            logfileName = os.path.join(model_home, year, sensor, logfile)
            os.chdir(os.path.dirname(solverName))
            cmd = "~/caffe/build/tools/caffe train --solver={} 2>&1 | tee {}".format( solverName, logfileName)    
            os.system(cmd)
            
            if os.path.isdir(os.path.dirname(logfileName)):
                  print "Log folder is already exist."
            else:
                print "Make log directory : "
                os.mkdir(os.path.dirname(logfileName))
              #%%
            logLocation = os.path.dirname(logfileName) 
              #%%   
            makeCSVCmd = "python /home/park/caffe/tools/extra/parse_log.py {} {}".format(logfileName, logLocation)
            os.system(makeCSVCmd)
            
              #%% 
            trLogTxt = logfileName+".train"
            teLogTxt = logfileName+".test"
            #teLogTxt = "/home/park/data/LIVDETECT/Model/LivDet2011/BiometrikaTrain/log/twoNoval.log.test"
            
            train_log = pd.read_csv(trLogTxt)
            test_log = pd.read_csv(teLogTxt)
            fig, ax1 = plt.subplots(figsize=(15, 10))
            ax2 = ax1.twinx()
            ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
            ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
            ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r')
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('train loss')
            ax2.set_ylabel('test accuracy')
            fig.savefig(logfileName+".png")
  
else:
  model_home = options.model_home
  year = options.year
  sensor = options.sensor
  solver = options.solver
  logfile = options.logfile

  solverName = os.path.join(model_home, year, sensor, solver)
  logfileName = os.path.join(model_home, year, sensor, logfile)

  os.chdir(os.path.dirname(solverName))

  #options.conti = True

  if options.conti:
      state = options.solverstate
      logfile = 'log/{}.log'.format(os.path.basename(state).split('.')[0])
      logfileName = os.path.join(model_home, year, sensor, logfile)
      solverState = os.path.join(model_home, year, sensor, state)
      cmd = "~/caffe/build/tools/caffe train --solver={} --snapshot={} 2>&1 | tee {}".format( solverName, solverState,logfileName)
      os.system(cmd)
  else:
      if os.path.isdir(os.path.dirname(logfileName)):
        print "Log folder is already exist."
      else:
        print "Make log directory : "
        os.mkdir(os.path.dirname(logfileName))
      cmd = "~/caffe/build/tools/caffe train --solver={} 2>&1 | tee {}".format( solverName, logfileName)    
      os.system(cmd)
      #%%
      

  #%%
  logLocation = os.path.dirname(logfileName) 
  #%%   
  makeCSVCmd = "python /home/park/caffe/tools/extra/parse_log.py {} {}".format(logfileName, logLocation)
  os.system(makeCSVCmd)

  #%% 
  trLogTxt = logfileName+".train"
  teLogTxt = logfileName+".test"
  #teLogTxt = "/home/park/data/LIVDETECT/Model/LivDet2011/BiometrikaTrain/log/twoNoval.log.test"

  train_log = pd.read_csv(trLogTxt)
  test_log = pd.read_csv(teLogTxt)
  fig, ax1 = plt.subplots(figsize=(15, 10))
  ax2 = ax1.twinx()
  ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
  ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
  ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r')
  ax1.set_xlabel('iteration')
  ax1.set_ylabel('train loss')
  ax2.set_ylabel('test accuracy')
  fig.savefig(logfileName+".png")

