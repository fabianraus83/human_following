#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:57:20 2021

@author: pinup1
"""
#used for human tracking
#tracking will be done for 1 human in the scene only

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 

import pyrealsense2.pyrealsense2 as rs
 
from PIL import Image as PILImage
from yolov4 import Detector

import numpy as np 
import time
import cv2
import math

angle1 = 15
angle2 = 30

path_darknet = "/home/ubuntu/YoloV4/darknet/"
#path_darknet = "/home/fabian/yolov4/darknet/"


def detect_humans(a):    
    img = PILImage.fromarray(a) # using frames directly without saving 
    print(type(img))
    d = Detector(meta_path= path_darknet + 'cfg/coco.data', lib_darknet_path= path_darknet + 'libdarknet.so', config_path= path_darknet + 'cfg/yolov4-tiny.cfg', weights_path= path_darknet + 'model_data/yolov4-tiny.weights',gpu_id = None)
#    d = Detector(config_path= path_darknet + 'cfg/yolov4-tiny.cfg', weights_path= path_darknet + 'model_data/yolov4-tiny.weights', gpu_id=1)    
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

def cartesiantopolar(x,z):
    radius = math.sqrt( x * x + z * z )
    theta = math.atan(x/z) #from center
    theta = 180 * theta/math.pi #get in degrees
    
    return theta, radius    

#def fuzzy(x,z):
def startstream():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    
    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    #print("config set")
    # Start streaming
    pipeline.start(config)
    
    #print("started pipe")

def getcoordinates(self):
#    self.pipeline = rs.pipeline()


# This call waits until a new coherent set of frames is available on a device
    # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
    frames = self.pipeline.wait_for_frames()
#    print("frames ready")
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:            
        print("error")
#        print(color_frame)
    
    color_image = np.asanyarray(color_frame.get_data()) #change to numpy to fit detector 
            
#    stream_vid = color_image.copy()
    # print("hi")
    
#    cv2.imshow('clone_stream', stream_vid)
    
    intrin = frames.profile.as_video_stream_profile().intrinsics
    pixel, humanconfi = detect_humans(color_image) #get coordinate of human based on yolov4
    print("pixel: ",pixel)
    if pixel == None:
        print("no human")
        self.human = False
#        human = False
    else:
        print(pixel[0],pixel[1])
        rspixel = (int(pixel[0]/608*640),int(pixel[1]/608*480))
        print("rspixel: ",rspixel)
        dist = depth_frame.get_distance(rspixel[0],rspixel[1])
        Point = rs.rs2_deproject_pixel_to_point(intrin, rspixel, dist)
        print("3D coordinate of this point is: ", Point)
        x,y,z = Point
        if z != 0:
            self.human = True
#            human = True
            self.theta, self.radius = cartesiantopolar(x,z)
            print("Polar coordinates: ", "\ntheta: ", self.theta, "\nradius(m): ",self.radius)
            
    k = cv2.waitKey(5) & 0xFF # Esc key to stop
    if k == 27:
#        self.exit = True
        return True

#try:
#
#        
#    while True:
#
#finally:
#    # Stop streaming
#    pipeline.stop()
    
class MinimalPublisher(Node):
      def __init__(self):
         super().__init__('minimal_publisher')
         self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
         timer_period = 1  # seconds
         self.timer = self.create_timer(timer_period, self.timer_callback)
         self.i = 0
         
         # Create a context object. This object owns the handles to all connected realsense devices
         self.pipeline = rs.pipeline()
        
         # Configure streams
         config = rs.config()
         config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
         config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
         #print("config set")
         # Start streaming
         self.pipeline.start(config)
        
        #print("started pipe")
#         print("start stream")
         self.human = bool()
         self.exit = bool()
         self.x = float(0)
         self.y = float(0)
         
      def timer_callback(self):
          try:
              getcoordinates(self)
              cmdmsg = Twist()
              if self.human:
                  cmdmsg.linear.x = float(self.radius)
                  cmdmsg.angular.z = float(self.theta)
                  self.x = float(self.radius)
                  self.y = float(self.theta)
                  self.publisher_.publish(cmdmsg)
                  self.get_logger().info('Publishing an cmd vel'+ str(self.human))
                  self.counter = 0
              else:
                  self.counter += 1
                  if self.counter > 5:
                      self.get_logger().info("published nothing, no human")
                      print("counter number: ", self.i)
                  else:
                      cmdmsg.linear.x = float(self.x)
                      cmdmsg.angular.z = float(self.y)
                      self.publisher_.publish(cmdmsg)
                      self.get_logger().info('Publishing an cmd vel using prev coordinates '+ str(self.human))
              if self.exit:
                  pipeline = rs.pipeline()
                  pipeline.stop()
                  print("ended")
              print('published', self.i)
              self.i+=1
              
          except KeyboardInterrupt:
              cmdmsg.linear.x = float(0)
              cmdmsg.angular.z = float(0)
              
def main(args=None):
    rclpy.init(args=args)
    print("initiating pub")
    print("started")
    time.sleep(5)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    print('spin done')
    minimal_publisher.destroy_node()

    rclpy.shutdown()
    print("shutdown")

if __name__ == '__main__':
    main()
