# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 03:00:35 2016

@author: park
"""

import sys
import os
import numpy as np

def ExtractPatch(img, stepSize):
    """
    description : Make patches divided into background and foreground
    ---------------------------------------    
    Input :
        img - grayscale fingerprint image
        stepSize : Patch size
    Return :
        bg - background patch images
        bgtl - background top left corners
    """
    hl = img.shape[0]/stepSize
    wl = img.shape[1]/stepSize
    hlist = np.array(range(hl))*stepSize
    wlist = np.array(range(wl))*stepSize
    bg =[]
    bgtl = []
#    result = np.zeros((hl,wl))
#    remaps = np.zeros((hl,wl))
    for i, h in enumerate(hlist):
        for j, w in enumerate(wlist):
            imgPatch =  img[h:h+stepSize,w:w+stepSize]
            bg.append(imgPatch)
            bgtl.append((h,w))
    return bg, bgtl

def load2011labels(sensorName):
    labels = {}
    if sensorName == "BiometrikaTest":
        labels ={"Live":0, "EcoFlex":1, "Gelatin":2, "Latex":3, "Silgum":4, "WoodGlue":5, "bg":6}
    elif sensorName == "DigitalTest":
        labels ={"Live":0, "Gelatin":1, "Latex":2, "Playdoh":3, "Silicone":4, "Wood Glue":5, "bg":6}
    elif sensorName == "ItaldataTest":
        labels ={"Live":0, "EcoFlex":1, "Gelatin":2, "Latex":3, "Silgum":4, "WoodGlue":5, "bg":6}
    elif sensorName == "SagemTest":
        labels ={"Live":0, "Gelatin":1, "Latex":2, "Playdoh":3, "Silicone":4, "Wood Glue":5, "bg":6}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels

    
def checkFake(hist, label):
    nFake = 0
    nLive = 0
    for key, value in hist.items():
        if key == label['bg']:
            continue
        if key == label['Live']:
            nLive = value
        else:
            nFake += value
    return nLive, nFake