#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:17:52 2021

@author: fabian
"""

import cv2

#camID = input("CamID: ")
#
#cap = cv2.VideoCapture(camID)
#
#ret, frame = cap.read()
#path = "/image/"
#i = 0
#
#while(True):
#   print('image available: ', ret)
#   i+=1
#   path = path + str(i) + ".png"
#   take_again = input("take again: ")
#   if take_again == "n":
#       break
#       
#
#cap.release()
#cv2.destroyAllWindows()

cam = cv2.VideoCapture(0)

image = cam.read()[1]

cv2.imwrite("image/image.png")

cv2.waitKey(0)
cv2.destroyAllWindows()