#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:25:17 2021

@author: fabian
"""

import cv2 as cv 

cam_list = []

def testDevice(source):
   cap = cv.VideoCapture(source)
   
   if cap is None or not cap.isOpened():
       print('Warning: unable to open video source: ', source)
       
   else:
       cam_list.append(source)
      
for i in range(4):
    testDevice(i)
    
print("\ncam list:")
print(cam_list)
