# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:17:32 2016

@author: park
"""

import numpy as np
import os
import pandas as pd
from pandas import Series, DataFrame
import pickle
from collections import Counter
from sklearn.metrics import confusion_matrix
import util_livdet as ut
import matplotlib.pyplot as plt
#import resultAnalizer as res
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

#%% 
def makeTwoClass(y_test, y_pred):
    new_target = []
    new_pred = []
    for i in xrange(len(y_test)):
        if y_test[i] == 0:
            new_target.append(0)
        else :
            new_target.append(1)
        if y_pred[i] == 0 :
            new_pred.append(0)
        else :
            new_pred.append(1)
    return new_target, new_pred
            

#%% Precision and Recall result
def precisionRecall(confmat):
    size = confmat.shape
    if size[0] is not size[1]:
        print("Shoud be Square Matrix")
        return None

    nominator = np.zero(size[0])
    for ind in xrange(size[0]):
        nominator[ind] = confmat[ind, ind]
    preDenom = np.sum(confmat, axis=0)
    recallDenom = np.sum(confmat, axis=1)
    precision = nominator/float(preDenom)
    recall = nominator/float(recallDenom)
    return precision, recall

#%% Draw Confusion Matrix
def plotPrecisionAndRecall(confmat):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    

#%%
# 레이블을 연도랑 센서 입력하면 뽑게 해주자
def loadLabelFromYear(year, sensor):
    if year == "LivDet2011":
        label = ut.load2011labels(sensor)
    elif year == "LivDet2013":
        label = ut.load2013labels(sensor)
    elif year == "LivDet2015":
        label = ut.load2015labels(sensor)
    else:
        print("You may write wrong year : ", year)
        label = None
    return label

#%% Very Important function : 이건 패치들 스코어들 모아서 보팅결과를 출력하고 그것을
# txt로 만드는 함수이다. 여기서 모든 분석이 시작된다
def patchToDetectionTextResult(testHome, year, sensor, folderName='Result', binary=False):
    finalPath = os.path.join(testHome, year, sensor)
    
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
    tfolders = label.keys()
    tfolders.remove('bg')
    if binary :
        label = {'Live':0, 'Fake':1, 'bg':2}
    for tfold in tfolders:
        resultF = os.path.join(finalPath, tfold)
        if os.path.isdir(resultF):
            result = []
            print(year+', ', tfold, ' is processing')
            fileList = ut.search(resultF, ['.pickle'])
            for pick in fileList:
               frame = pickToDataFrame(pickleLoader(pick), ind='pos')
               lindex = predictFromVote(frame, label)
               if lindex is None:
                print pick, " is poor."
               else:
                result.append([pick.split('/')[-1].split('.')[0], lindex])
            resultPath = os.path.join(finalPath, folderName)
            ut.makeBGFGFolder(resultPath)
            fileName = os.path.join(resultPath, tfold)
            listToTxt(result, fileName+'.txt', sel='pred')
        else:
            print(tfold, ' is not exist.')


#%% 폴더에서 값들 읽어서 Confusion matrix를 만든다
"""
# Input : 
    # hdirec:  이미지 별 결과가 들어있는 디렉토리
    # label : 레이블
# Output :
    # y_test : 타겟 레이블
    # y_pred : 예측결과
"""

#def makeClassTwo(confmat) :
    

def makePrediction(hDirec, label, binary=False):
#    hDirec = '/home/park/data/LIVDETECT/TestResult/LivDet2011/BiometrikaTrain/Result'
    resultfile = ut.search(hDirec, ['.txt'])
    y_test = []
    y_pred = []
    for rr in resultfile:
        labelname = os.path.basename(os.path.splitext(rr)[0])
        print(labelname+' is working')
        f = open(rr, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            if binary :
                if labelname == 'Live':
                    val = 0
                else:
                    val = 1
                y_test.append(val)
                predict = line.split(' ')[-1]
                y_pred.append(int(predict))
            else:    
                y_test.append(label[labelname])
                predict = line.split(' ')[-1]
                y_pred.append(int(predict))
#    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    return y_test, y_pred

#%% 타겟과 예측된 값 받아서 Confusion matrix를 만든다
def makeConfusionMatrix(y_test, y_pred):
#    hDirec = '/home/park/data/LIVDETECT/TestResult/LivDet2011/BiometrikaTrain/Result'
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    return confmat

#%% Calcuate precision    
def makePrecision(y_test, y_pred):
    return precision_score(y_true=y_test, y_pred = y_pred)

#%% Calculate recall
def makeRecall(y_test, y_pred):
    return recall_score(y_true=y_test, y_pred = y_pred)    
    
#%% Calculate f1 score
def makeF1Score(y_test, y_pred):
    return f1_score(y_true=y_test, y_pred = y_pred)        
    
#%% This is pickle file Loader
#============================================================
    
def listToTxt(result, fileName, sel='pred'):
    if sel == 'pred':
        f = open(fileName, 'w')    
        for item in result:
            line = "{0} {1}{2}".format(item[0], item[1],'\n')
            f.write(line)
        f.close()
    if sel == 'top':
        f = open(fileName, 'w')    
        for item in result:
            line = "{0} {1} {2} {3}{4}".format(item[0], item[1], item[2], item[3], '\n')
            f.write(line)
        f.close()

def pickleLoader(fineName):
    pklFile = open(fineName)
    result = pickle.load(pklFile)
    pklFile.close()
    return result
#============================================================

def MultiRecallPrecision(confmat):
    nomi = np.diag(confmat)
    predenom = np.sum(confmat, axis = 0)
    recalldenom = np.sum(confmat, axis = 1)
    recall = nomi/recalldenom.astype(float)
    precision = nomi/predenom.astype(float)
    return precision, recall
    

#%% make list to DataFrame
# result : pickle을 로드해서 저장한 파일
# ind : index로 만들 컬럼
#============================================================\
def pickToDataFrame(result, ind='pos'):
    position = []
    labels = []
    score = []
    for value in result:
        position.append(value[0])
        labels.append(value[1])
        score.append(value[2])
    if ind == 'pos':
        data = {'label':labels, 'score':score}
        frame = DataFrame(data, index=position)
    elif ind == 'label':
        data = {'pos':position, 'score':score}
        frame = DataFrame(data, index=labels)
    elif ind == 'score':
        data = {'pos':position, 'label':labels}
        frame = DataFrame(data, index=score)
    else:
        print "ind parameter is wrong : ", ind
    return frame
#============================================================
    
#%% Predcit from patches
def predictFromVote(frame, label, top=True):
    relabel = frame['label']
    counted = Counter(relabel)
    if label['bg'] in counted: del counted[label['bg']]
    if top is True:
        try:
            return counted.most_common(1)[0][0]
        except:
            return None
    return counted
    
#def gatherLabels(frame, label, top=True):
#    relabel = frame['label']
#    counted = Counter(relabel)
#    if label['bg'] in counted: del counted[label['bg']]
#    if top is True:
#        return counted.most_common(1)[0][0]
#    return counted

#homeFolder = '/home/park/data/LIVDETECT/TestResult/LivDet2011/BiometrikaTrain/Live'
#fileName = '7_1.pickle'
#
#result = pickleLoader(os.path.join(homeFolder, fileName))


