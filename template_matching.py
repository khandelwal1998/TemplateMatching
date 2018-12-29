# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:52:41 2018

@author: abhishek
"""

import cv2
import numpy as np
img=cv2.imread("traffic.jpg") #Loading the main image
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
temp=cv2.imread("temp.jpg",0) #Loading the template
w,h=temp.shape[::-1]
res=cv2.matchTemplate(img_gray,temp,cv2.TM_CCOEFF_NORMED) #Template matching
thresh=0.66 #Setting the threshold value. Only those location with probability above thresh is needed
loc=np.where(res>=thresh)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(255,25,0),2) #Marking all those points where the image is detected
cv2.imshow("Image",img)