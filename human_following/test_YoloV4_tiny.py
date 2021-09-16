#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:13:48 2021

@author: fabian
"""

from PIL import Image as PILImage
from yolov4 import Detector

import numpy as np 

    
path_darknet = "/home/fabian/yolov4/darknet/"

#def detect_humans(a):    
#    img = PILImage.fromarray(a) # using frames directly without saving 
#    print(type(img))
#    d = Detector(config_path= path_darknet + 'cfg/yolov4-tiny.cfg', weights_path= path_darknet + 'model_data/yolov4-tiny.weights', gpu_id=1)
#    img_arr = np.array(img.resize((d.network_width(), d.network_height())))
#    detections = d.perform_detect(image_path_or_buf=img_arr, show_image=True)
#    peoplenumber = 0
#    maxconfi = 0
#    humancoordinate = tuple()
#    for detection in detections:
#        box = detection.left_x, detection.top_y, detection.width, detection.height
#        print(f'{detection.class_name.ljust(10)} | {detection.class_confidence * 100:.1f} % | {box}') 
#        if detection.class_name == 'person'and detection.class_confidence * 100 >= 50:
#            peoplenumber += 1
#            if detection.class_confidence >= maxconfi:
#                centrecoordinate = (int(detection.left_x + 0.5 * detection.width), int(detection.top_y + 0.5 * detection.height))
#                maxconfi = (int(detection.class_confidence*100))
##coordinate is based on resized image
#    
#    if peoplenumber == 0:
#        humancoordinate = None
#    
#    elif peoplenumber > 0 :
#        maxconfi = detection.class_confidence
#        humancoordinate = centrecoordinate
#        
#    return humancoordinate, maxconfi  

def detect_humans(a):    
    img = PILImage.open('image/image.png')    
    print(type(img))
    d = Detector(config_path= path_darknet + 'cfg/yolov4-tiny.cfg', weights_path= path_darknet + 'model_data/yolov4-tiny.weights', gpu_id=1)
    img_arr = np.array(img.resize((d.network_width(), d.network_height())))
    detections = d.perform_detect(image_path_or_buf=img_arr, show_image=True)
    peoplenumber = 0
    maxconfi = 0
    humancoordinate = tuple()
    for detection in detections:
        box = detection.left_x, detection.top_y, detection.width, detection.height
        print(f'{detection.class_name.ljust(10)} | {detection.class_confidence * 100:.1f} % | {box}') 
        if detection.class_name == 'person'and detection.class_confidence * 100 >= 50:
            peoplenumber += 1
            if detection.class_confidence >= maxconfi:
                centrecoordinate = (int(detection.left_x + 0.5 * detection.width), int(detection.top_y + 0.5 * detection.height))
                maxconfi = (int(detection.class_confidence*100))
#coordinate is based on resized image
    
    if peoplenumber == 0:
        humancoordinate = None
    
    elif peoplenumber > 0 :
        maxconfi = detection.class_confidence
        humancoordinate = centrecoordinate
        
    return humancoordinate, maxconfi  