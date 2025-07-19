import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def histogram_equal():
    root=os.getcwd()
    imgPath=os.path.join(root,r'D:\SEM7PROJ\fracturedetection\backend\clahe\clahetest1.jpg')
    img=cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
    hist=cv.calcHist([img],[0],None,[256],[0,256])
    cdf=hist.cumsum()
    cdfnorm=cdf * float(hist.max()) / cdf.max()
    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfnorm, color='b')
    plt.xlabel('Pixel intensity')
    plt.ylabel('No of pixels')

    equImg=cv.equalizeHist(img)
    equiHist=cv.calcHist([equImg],[0],None,[256],[0,256])
    equiCdf=equiHist.cumsum()
    equiCdfNorm=equiCdf * float(equiHist.max()) / equiCdf.max()
    plt.subplot(232)
    plt.imshow(equImg, cmap='gray')
    plt.subplot(235)
    plt.plot(equiHist)
    plt.plot(equiCdfNorm, color='b')
    plt.xlabel('Pixel intensity')
    plt.ylabel('No of pixels')
    plt.show()


if __name__=='__main__':
    histogram_equal()