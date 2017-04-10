# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import cv2
import shutil
"""
    이 유틸모듈은 대부분의 경우 Caffe를 이용하여 프로그래밍 할 수 있도록 돕는
    유틸리티 프로그램들이 위치하게 됩니다. 
"""

def makeRootDir(specific, dataset):
    """ 
        Description : return livdet's folder paths
        Input :
            - specific : foldername having livdet's
            - dataset : selector among 2011, 2013 and 2015
        Output :
            - Livdet abosolute path
    """
    if os.path.isdir(specific):
        dataName ='LivDet'
        if dataset in ['2011', '2013', '2015']:
            dataName = dataName+dataset
            return os.path.join(specific, dataName)
        else:
            print("Only support LivDet2011, 2013 and 2015.")
            print("Default folder is LivDet2011.")
            return os.path.join(specific, 'LivDet2011')
    else:
        print("Can not find %s folder" % specific)
        
def folderFiles(rootDir, exts):
    """
    Desrciption : find recursively files and included folders 
    with some extensions
    ---------------------------------------    
    Input :
        rootDir - root directory 
        exts - related extensions name (list)
    Return :
        directory : directory names
        imglist : list of list files names according to directory
    """
    imglist = []
    directory = []
    for rootdir, dirnames, filenames in os.walk(rootDir):
        imgs = []
        if len(filenames) > 0:        
            for ff in filenames:
                if ff[-4:] in exts:
                    imgs.append(ff)
            if len(imgs) > 0:
                directory.append(rootdir)
                imglist.append(imgs)
    return directory, imglist

#==============================================================================
#  In belows, many of them are only related to making Patches.
#==============================================================================

def makeDestFolder(oringin, direc):
    """
    description : make desination folder
    ------------------------------------------
    Input :     
        origin - original destination folder name
        direc - folders having original images
    Output :
        destFolder - destination folder name having same 
        structure of input folder
    """
    sep = os.path.basename(oringin)
    destFolder = []
    for ff in direc:
        last = ff.split(sep)[-1]
        newDest = os.path.join(oringin+last)
        destFolder.append(newDest)
        if os.path.isdir(newDest): print "Directory exists : "+newDest
        else: 
            os.makedirs(newDest)
            print "Make Dir : " + newDest
    return destFolder

def makeBGFGFolder(Folder):
    """
    decription : make new folder if there is no same folder name
    -----------------------------------------
    Input :
        Folder - new folder name
    """
    if os.path.isdir(Folder):
        print "-- Exist : " + Folder
    else:
        os.mkdir(Folder)
        print "-- Make : " + Folder

def makeTxtName(baseName, bgName, FolderName):
    """
    decription : make text files represent patchname and it's corresponding 
    original image name
    -----------------------------------------
    Input :
        baseName - base name
        bgName - indicators differenciating bg and fg
        FolderName - destination text file
    return :
        bgtxtName - textfile name including paths
    """
    bgtxt = os.path.basename(baseName)+'_'+bgName+'.txt'
    bgtxtName = os.path.join(FolderName, bgtxt)
    return bgtxtName

def makePatchData(img, seg, stepSize, ratio):
    """
    description : Make patches divided into background and foreground
    ---------------------------------------    
    Input :
        img - grayscale fingerprint image
        seg - segmented binary image
        stepSize : Patch size
        ratio : ratio of segmented area per patch size
    Return :
        bg - background patch images
        fg - foreground patch images
        bgtl - background top left corners
        fgtl - foreground top left corners
    """
    hl = img.shape[0]/stepSize
    wl = img.shape[1]/stepSize
    hlist = np.array(range(hl))*stepSize
    wlist = np.array(range(wl))*stepSize
    bg =[]
    fg = []
    bgtl = []
    fgtl = []
#    result = np.zeros((hl,wl))
#    remaps = np.zeros((hl,wl))
    for i, h in enumerate(hlist):
        for j, w in enumerate(wlist):
            imgPatch =  img[h:h+stepSize,w:w+stepSize]
            segPatch = seg[h:h+stepSize,w:w+stepSize]
            rate = np.count_nonzero(segPatch)/float((stepSize*stepSize))
#            result[i,j] = rate
            if rate > ratio:
                fg.append(imgPatch)
                fgtl.append((h, w))
            else:
                bg.append(imgPatch)
                bgtl.append((h,w))
    return bg, fg, bgtl, fgtl
    
def patchSave(patches, location, folder, png_params, img, files):
    """
        description : Save patches and make filelist
        (it need to change because of files parameter is not adequate.)
        -----------------------------------------------------
        Input :
            patches - patches
            location - top left corner of the patches
            folder - saving location
            png_params - compression level for png file 
            files - file (must be opened)
    """
    for bi, bb in enumerate(patches):
        hh = location[bi][0]
        ww = location[bi][1]
        iiName = str(bi)+'_'+str(hh)+'x'+str(ww)+'.png'
        bNames = os.path.join(folder, iiName)
        cv2.imwrite(bNames, bb, png_params)
        files.write(iiName+','+img+'\n')

#==============================================================================
# 이 아래 구간은 생성된 텍스트 파일을 다루는 함수들이 존재한다.
#==============================================================================
import shutil
from pandas import Series, DataFrame
import pandas as pd

class txtPaser():
    """ 
        txt파일을 읽은 후 pandas series로 변환하는 class이다
        잘 활용해 보려 했지만 거의 쓸일이 없는 것 같다
    """
    def __init__(self, txtName, sampleNum, label=[]):
        # txt file save
        self.txtName = txtName
        # text to padas DataFrame
        self.pdTable = pd.read_csv(txtName, header=None)
        if sampleNum is not None:
            self.pdTable = self.pdTable[:sampleNum]
        # save labels
        self.label = label
    
    def selectOne2Lists(self, sensorName):
        # sensorName과 일치하는 녀석들만 반환하는 함수 
        sensors = []
        for line in self.pdTable.values:
            sensor = line[0].split(os.sep)
            if sensorName in sensor:
                sensors.append(line[0])
            else:
                continue
        return sensors
            
def separateSpecificImages(inList, keyword, label):
    """
        keyword에 해당하는 이미지들에 label을 부여한 후 저장 
    """
    specific = []
    others = []
    for img in inList:
        fragment = img.split(os.sep)
        if keyword in fragment:
            specific.append(img+' '+str(label))
        else:
            others.append(img)
    return specific, others

def loadSensorList(year):
    if year == "LivDet2011":
        return ["BiometrikaTrain", "DigitalTrain", "ItaldataTrain","SagemTrain"]
    elif year == "LivDet2013":
        return ["BiometrikaTrain", "CrossMatchTrain", "ItaldataTrain","SwipeTrain"]
    elif year == "LivDet2015":
        return ["CrossMatch", "Digital_Persona", "GreenBit", "Hi_Scan", "Time_Series"]
    else:
        print "There is no %s Dataset" % year
        return False        

def loadLabelFromYear(year, sensor):
    if year == "LivDet2011":
        label = load2011labels(sensor)
    elif year == "LivDet2013":
        label = load2013labels(sensor)
    elif year == "LivDet2015":
        label = load2015labels(sensor)
    else:
        print("You may write wrong year : ", year)
        label = None
    return label
           

def load2011labels(sensorName):
    labels = {}
    if sensorName == "BiometrikaTrain":
        labels ={"Live":0, "EcoFlex":1, "Gelatin":2, "Latex":3, "Silgum":4, "WoodGlue":5, "bg":6}
    elif sensorName == "DigitalTrain":
        labels ={"Live":0, "Gelatin":1, "Latex":2, "Playdoh":3, "Silicone":4, "Wood Glue":5, "bg":6}
    elif sensorName == "ItaldataTrain":
        labels ={"Live":0, "EcoFlex":1, "Gelatin":2, "Latex":3, "Silgum":4, "WoodGlue":5, "bg":6}
    elif sensorName == "SagemTrain":
        labels ={"Live":0, "Gelatin":1, "Latex":2, "Playdoh":3, "Silicone":4, "Wood Glue":5, "bg":6}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels
    
def load2013labels(sensorName):
    labels = {}
    if sensorName == "BiometrikaTrain":
        labels ={"Live":0, "Ecoflex":1, "Gelatin":2, "Latex":3, "Modasil":4, "WoodGlue":5, "bg":6}
    elif sensorName == "CrossMatchTrain":
        labels ={"Live":0, "BodyDouble":1, "Latex":2, "Playdoh":3, "WoodGlue":4, "bg":5}
    elif sensorName == "ItaldataTrain":
        labels ={"Live":0, "Ecoflex":1, "Gelatine":2, "Latex":3, "Modasil":4, "WoodGlue":5, "bg":6}
    elif sensorName == "SwipeTrain":
        labels ={"Live":0, "BodyDouble":1, "Latex":2, "Playdoh":3, "WoodGlue":4, "bg":5}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels
    
def load2015labels(sensorName):
    labels = {}
    if sensorName == "CrossMatch":
        labels ={"Live":0, "Body Double":1, "Ecoflex":2, "Playdoh":3, "bg":4}
    elif sensorName == "Digital_Persona":
        labels ={"Live":0, "Ecoflex 00-50":1, "Gelatine":2, "Latex":3, "WoodGlue":4, "bg":5}
    elif sensorName == "GreenBit":
        labels ={"Live":0, "Ecoflex 00-50":1, "Gelatine":2, "Latex":3, "WoodGlue":4, "bg":5}
    elif sensorName == "Hi_Scan":
        labels ={"Live":0, "Ecoflex 00-50":1, "Gelatine":2, "Latex":3, "WoodGlue":4, "bg":5}
    elif sensorName == "Time_Series":
        labels ={"Live":0, "Body Double":1, "Ecoflex":2, "Playdoh":3, "bg":4}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels

def saveTableToTxt(Container, destFolder):
    #  txt 파일로 저장 
    for key, value in Container.items():
        fileName = key+".txt"
        Series(value).to_csv(os.path.join(destFolder, fileName), index=False, header=False)
        
def saveListToTxt(Container, destFolder, fileName, force=False):
    #  txt 파일로 저장
    fName = os.path.join(destFolder, fileName)
    if (os.path.isfile(fName)) and (not force):
        print "The same name file is exist"
    else:
        dfileName = os.path.join(destFolder, fName)
        f = open(dfileName, 'w')
        f.writelines(Container)
        f.close
#        Series(Container).to_csv(os.path.join(destFolder, fName), index=False, header=False)



def makeStat(txtTotal, fold=5):
    # Cross validation에 필요한 인덱스들을 반환한다
    # 맨처음엔 그냥 이미지 갯수를 넣어봤다 
    temp = {}
    for key, value in txtTotal.items():
        tt = []
        step = len(value)//fold
        for f in range(5):
            if f == 0: tt.append(len(value))
            else: tt.append(f*step)
        temp[key] = tt
    return temp

def makeCrossValidation(stat, txtTotal):
    # Cross validation단위로 데이터를 저장함 
    crossValContainer = {}
    for label, data in txtTotal.items():
        data = np.array(data)
        val = []
        train = []
        last = len(stat[label])-1
        for i, index in enumerate(stat[label]):
            mask = np.zeros(data.shape, dtype=bool)
            if i==0:
                mask[:stat[label][i+1]] = True
            elif i==last:
                mask[index:] = True
            else:
                mask[index:stat[label][i+1]] = True
            val.append(data[mask].tolist())
            train.append(data[~mask].tolist())
        crossValContainer[label] = {}
        crossValContainer[label]['val'] = val
        crossValContainer[label]['train'] = train
    
    return crossValContainer

def reduceNum(txtTotal, label, start, end):
    # 특정 label의 start와 end만큼으로 크기를 줄인다 
    txtTotal[label] = txtTotal[label][start:end]
    return txtTotal
    
def valTrainMerge(trValContainer, fold):
    val = [[] for i in range(fold)]
    train = [[] for i in range(fold)]
    for i, value in trValContainer.items():
        for kk in range(fold):
            val[kk] += value['val'][kk]
            train[kk] += value['train'][kk]
    return val, train


def saveTrainVal(val, train, destFolder, fileName):
    # filename[0] is validation and filename[1] is training        
    if os.path.isdir(destFolder):
        print("Exist folder")
    else:
        print("Make %s folder." % os.path.basename(destFolder))
        os.mkdir(destFolder)
    for i, valid in enumerate(val):
        saveListToTxt(valid, destFolder, fileName[0]+str(i)+'.txt')
        saveListToTxt(train[i], destFolder, fileName[1]+str(i)+'.txt')

def shufflingSet(val):
    newVal = []
    for v in val:
        newVal.append(shuffling(v))
    return newVal
        
def shuffling(data):
    permutation = np.random.permutation(len(data))
    data = np.array(data)
    data = data[permutation]
    return data.tolist()

def folderFilter(direc, listFiles, regs):
    removal = np.ones(len(direc), dtype=bool)
    for ii, dirs in enumerate(direc):
        if os.path.basename(dirs) not in regs:
            removal[ii] = False
        else:
            removal[ii] = True
    npDirec = np.asarray(direc)
    npListFiles = np.asarray(listFiles)
    npDirec = npDirec[removal]
    npListFiles = npListFiles[removal]
    return npDirec, npListFiles

def copytextFiles(orFolderName, trFolderName, txtFolder, noCopy=True):    
    direc, listFiles = folderFiles(orFolderName, ['.txt'])
    direc, listFiles = folderFilter(direc, listFiles, ['bg','fg'])    
    if noCopy:
        return direc, listFiles
        
    for ii in range(len(direc)):    
        sensorFolder = direc[ii].split(trFolderName)
        sensorName = sensorFolder[-1].split('/')[1]
        textFolder = os.path.join(sensorFolder[0], trFolderName, 
                                  sensorName, txtFolder)
        makeBGFGFolder(textFolder)
        orgImgPath = os.path.join(direc[ii], listFiles[ii][0])
        detImgPath = os.path.join(textFolder, listFiles[ii][0])
        shutil.copy(orgImgPath, detImgPath)        

#==============================================================================
# 
#==============================================================================
def makeTxtListFiles(dataset):
    trFolderName = "Training"
    patchName = "Patch"
     
    txtFolder = 'txtFolder'
    orFolderName = os.path.join(dataset, patchName)
     
    direc, listFiles = copytextFiles(orFolderName, trFolderName, txtFolder)

    fileList = []
    for index, folder in enumerate(direc):
        filepath = os.path.join(folder, listFiles[index][0])
        fileList.append(filepath)
    print fileList

    txtList = os.path.join(orFolderName, 'txtList.txt')
    f = open(txtList, 'w')
    for line in fileList:
        f.write(line+"\n")
    f.close()

def maketx32and64List(rootDir):
#    trFolderName = "Training"
    patchName = "Patch"
    txtList = "txtList.txt"
    step32 = '32x32'
    step64= '64x64'
#    txtFolder = 'txtFolder'
    fileName = os.path.join(rootDir, patchName, txtList)
    
    fileList = pd.read_csv(fileName, names=['txts'])['txts']    
    
#    with open(fileName, "r") as f:
#        fileList = f.readlines()
    newPatch = []
    for fname in fileList:
        filetxt = fname.strip()
        result = pd.read_csv(filetxt, names=['patch','origin'])
        if result is None:
            print "Fail to load txt file"
        else:
            dirName = os.path.dirname(filetxt)
            baseName = dirName.split(os.sep)[-1]
            patches = result['patch']
            newPatch.append((dirName+'/')+patches)
    resultPatche = pd.concat(newPatch)

    tx32 = []
    tx64 = []
#    sample = resultPatche[:10]
    for text in resultPatche:
        if text.split(patchName)[1].split(os.sep)[1] == step32:
            tx32.append(text)
        elif text.split(patchName)[1].split(os.sep)[1] == step64:
            tx64.append(text)
        else:
            pass

    txt32 ='t32x32.txt'
    txt64 = 't64x64.txt'
    tx32name = os.path.join(rootDir, patchName, txt32)
    tx64name = os.path.join(rootDir, patchName, txt64)
    tx32 = Series(tx32)
    tx64 = Series(tx64)
    tx32.to_csv(tx32name, index=False, header=False)
    tx64.to_csv(tx64name, index=False, header=False)

def search(dirname, extension):
    """    
    desrciption : find files of directory with extension
    ---------------------------------------    
    Input :
        dirname - root directory 
        extension - related extensions name (list)
    Return :
        filelist : file list of dirname folder
    """
    filelist = []
    filenames = os.listdir(dirname)    
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext in extension:
            filelist.append(full_filename)
    return filelist

def labelingAndmakeTxt(rootDir, dataset, sensor, stepSize):
     fileList = "t"+stepSize+".txt"
#     sensor = "Time_Series"
     lists = txtPaser(os.path.join(rootDir, fileList),None)
     
     destDir = os.path.join(rootDir, stepSize,"Training", sensor)
     if dataset is "LivDet2011":
         labels = load2011labels(sensor)
     elif dataset is "LivDet2013":
         labels = load2013labels(sensor)
     elif dataset is "LivDet2015":
         labels = load2015labels(sensor)
     else:
         print "There is no available label files"

     Bio = lists.selectOne2Lists(sensor)

     Container={}
     Container["bg"], others = separateSpecificImages(Bio, "bg", labels["bg"])

     for key, lab in labels.items():
         if Container.get(key) is None:
             Container[key], others = separateSpecificImages(others, key, lab)
         
 
     destFolder = os.path.join(destDir, "ListTxt")
     makeBGFGFolder(destFolder)
     
     saveTableToTxt(Container, destFolder)
     
def makeCrossValSet(baseDir, dataset, stepSize, destF, fileName, fold):
    rootDir = baseDir+dataset+"/Patch/"+stepSize+"/Training"
    ListTxt = "ListTxt"
    sensors = loadSensorList(dataset)
 #    for sensor in sensors:
    for sensor in sensors:    
        txtFolder = os.path.join(rootDir, sensor, ListTxt)
        print("======Start %s sensor.=======" % sensor )
        textFiles = search(txtFolder, ['.txt'])
        if dataset == "LivDet2011":
            labels = load2011labels(sensor)
        elif dataset == "LivDet2013":
            labels = load2013labels(sensor)
        elif dataset == "LivDet2015":
            labels = load2015labels(sensor)
        txtTotal = {}
        for text in textFiles:
            material = os.path.basename(text).split('.')[0]
            with open(text, 'r') as f:
                txtTotal[material] = f.readlines()
    #        txtTotal[material] = pd.read_csv(text)
        txtTotal = reduceNum(txtTotal, 'bg', 0, len(txtTotal["Live"]))
        stat = makeStat(txtTotal, fold)
        crossValContainer = makeCrossValidation(stat, txtTotal)
        
        destFolder = os.path.join(rootDir, sensor, destF)
        val, train = valTrainMerge(crossValContainer, fold)
        
        val = shufflingSet(val)
        train = shufflingSet(train)
        
        saveTrainVal(val, train, destFolder, fileName)

#==============================================================================
# 실제로 텍스트를 다루고 조정하는 작업을 하는 함수들은 여기부터 시작하도록 하자     
#==============================================================================
def makeDataForLMDB(saveFolder, sensorName, textList, saveTxt):
    """
        textList파일에 위치한 패치들을 saveFolder로 복사하는데, 이름은 물질종류
        즉, Live, Latex 처럼 유지시키도록 한
    """
    # Copy test
    if os.path.isdir(saveFolder):
        print saveFolder + " is already exist."
    else:
        os.makedirs(saveFolder)
    newFileList = []
    with open(textList, 'r') as f:
        texts = f.readlines()
#    texts = texts[:10]
    for te in texts:
        orImgName, label = te.split(' ')
#        oldName = te.split(sensorName)[1].split(' ')[0]
        newName = te.split(sensorName)[1].split(' ')[0][1:].replace("/","_")
        newFileList.append(newName+' '+label)
        shutil.copy(orImgName, os.path.join(saveFolder, newName))

    with open(saveTxt, 'w') as f:
        for newName in newFileList:
            f.write(newName)

        
#    shutil.copy("newFile.txt",saveFolder)
    
    

def textSeparation(textFile, sensorName, saveLocation):
    """
        Description : Load text file and return Root folder name and save
        텍스트 파일을 받아서 공통영역을 제거 한 후 다시 저장한다. CAFFE의 lmdb
        만들때 이렇게 해줘야 할 수 있기때문에 만들어 본다.
        Input :
            - textFile : text file name
            - sensorName : separator name
            - saveLocation : save fileName
        return : True or False
    """
    with open(textFile, 'r') as f:
        texts = f.readlines()
#    print texts[:10]
#    newText = []
    with open(saveLocation, 'w') as f:
        for te in texts[:10]:
            splited = te.split(sensorName)[1]
            f.write(splited)
            
        
     

if __name__ == "__main__":
#==============================================================================
#     Test : makeRootDir function
#==============================================================================
    my_livdet = "/home/park/mnt/DBs/FAKE/lvedet/Old_Datasets"    
    rootDir = makeRootDir(my_livdet, '2016')
    print rootDir

