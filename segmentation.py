import cv2
import numpy as np
import os

def weonjinSegmentation(mInput):
    """
    description : segmentation of fingerprint image (binary)
    It's a Pyhon version of Weonjin Kim's segmentatin method
    ---------------------------------------    
    Input :
        mInput - gray scale ndarray
    Return :
        mTemp2 - segmented image (binary)
    """
    coh = np.zeros(mInput.shape, dtype='f4')
    Gxx = np.zeros(mInput.shape, dtype='f4')
    Gyy = np.zeros(mInput.shape, dtype='f4')
    eisum = np.zeros(mInput.shape, dtype='f4')
    
    kernel1 = np.array([[-1],[1]])
    mOutput1 = cv2.filter2D(mInput, cv2.CV_8U, kernel1)
    mOutput2 = cv2.filter2D(mInput, cv2.CV_8U, kernel1.reshape(1,2))
    mOutput1 = mOutput1/255.
    mOutput2 = mOutput2/255.
    
    Gyy = mOutput1**2
    Gxx = mOutput2**2
    Gxy = mOutput1*mOutput2

    nWinSize = 5
    nWinHalf = nWinSize/2
#    h, w = 0, 0
    i = nWinHalf
    vcoh = []
    veisum = []
    while i < (mInput.shape[0]-nWinHalf):
        j = nWinHalf
        while j < (mInput.shape[1] - nWinHalf):
            mNumMatx = Gxx[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            sumx = mNumMatx.sum()
            mNumMaty = Gyy[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            sumy = mNumMaty.sum()
            mNumMatxy = Gxy[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            sumxy = mNumMatxy.sum()
            mNumMat = coh[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            eigenmax = np.abs((sumx + sumy + np.sqrt((sumx - sumy)**2 + 4 * sumxy*sumxy))) / 2.
            eigenmin = np.abs((sumx + sumy - np.sqrt((sumx - sumy)**2 + 4 * sumxy*sumxy))) / 2.
            eigensum=eigenmax+eigenmin
            eisum[i,j] = eigensum*2
            mNumMatsum = eisum[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            if(eigenmax+eigenmin == 0):
                coh[i,j] = 0
            else:
                coh[i,j] = (eigenmax-eigenmin)/(eigenmax+eigenmin)
            mNumMat[:] = coh[i,j]
            mNumMatsum[:] = eigensum*2
            vcoh.append(coh[i,j])
            veisum.append(eisum[i,j])
            j = j+nWinSize
        i = i+ nWinSize
    mcoh = np.array(vcoh)
    meisum = np.array(veisum)
    mcoh *= 255
    meisum *= 255
    mcoh = mcoh.astype(np.uint8)
    meisum = meisum.astype(np.uint8)
    thcoh = cv2.threshold(mcoh, 0, 255, cv2.THRESH_OTSU)[0]
    theisum = cv2.threshold(meisum, 0, 255, cv2.THRESH_OTSU)[0]
    mTemp = np.copy(coh)
    mTemp2 = np.copy(eisum)
    #%%
    mask2 = (mTemp>(thcoh*2.)/(255*3)) & (mTemp2>(theisum/(255.*2)))
    mask1 = (mTemp>(thcoh)/(255.*2)) & (mTemp2>((theisum*2)/(255.*3)))
    mask3 = ~(mask2 | mask1)
    mTemp2[mask2] = 255
    mTemp2[mask1] = 255
    #%%
    mTemp2 [mask3] = 0
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel5 =cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    
    mTemp2 = cv2.dilate(mTemp2, kernel5, iterations = 1)
    mTemp2 = cv2.morphologyEx(mTemp2, cv2.MORPH_CLOSE, kernel4,iterations = 2)
    mTemp2 = cv2.morphologyEx(mTemp2, cv2.MORPH_OPEN, kernel3,iterations = 10)
    return mTemp2