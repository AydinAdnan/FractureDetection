import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def highlight_fractures():
    root = os.getcwd()
    imgPath = os.path.join(root, r'D:\SEM7PROJ\fracturedetection\backend\clahe\clahetest1.jpg')
    
    # Read image in grayscale
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found.")
        return

    # 1. Histogram Equalization (basic global)
    equ_img = cv.equalizeHist(img)
    
    # 2. CLAHE (local histogram equalization)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    clahe_img = clahe.apply(img)
    plt.figure(figsize=(12, 8))



    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original X-ray')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(equ_img, cmap='gray')
    plt.title('Histogram Equalized')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(clahe_img, cmap='gray')
    plt.title('CLAHE Enhanced')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    highlight_fractures()
