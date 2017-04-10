# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:10:03 2016

@author: park
"""

#import os
import pandas as pd
import matplotlib.pyplot as plt

trLogTxt = "/home/park/data/LIVDETECT/Model/LivDet2011/BiometrikaTrain/log/twoNoval.log.train"
teLogTxt = "/home/park/data/LIVDETECT/Model/LivDet2011/BiometrikaTrain/log/twoNoval.log.test"

train_log = pd.read_csv(trLogTxt)
test_log = pd.read_csv(teLogTxt)
_, ax1 = plt.subplots(figsize=(15, 10))
ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')