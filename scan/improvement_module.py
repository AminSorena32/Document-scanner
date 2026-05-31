import cv2 as cv
import numpy as np
def Improvement(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(gray,(3,3),0) 
    thresh=cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,17,6)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(2,2))
    thresh=cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=1)
    return thresh
