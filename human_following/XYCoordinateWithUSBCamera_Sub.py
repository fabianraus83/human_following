#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 18:03:41 2021

@author: fabian
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point 
from cv_bridge import CvBridge
import cv2
import numpy as np

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(Point, 'XY', self.listener_callback, 10) ###### here
        self.subscription
        self.i = 0
        
    def listener_callback(self,msg):
        self.i += 1
        print("listener iteration: ", self.i)
        self.get_logger().info('sub done. interation: ')
        print("x: {} \ny: {}".format(msg.x,msg.y))
        self.get_logger().info(str(msg.x))
        self.get_logger().info(str(msg.y))
        
def main(args=None):

    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

#def main(args=None):
#    pic1 = (return_array(subfolder, "c1.png"))
#    pic2 = (return_array(subfolder, "c1.png"))
#    msg = BatchPic()
#    create_batch(msg,pic1,pic2,1)
##    msg.pic1 = pic1
##    msg.pic2 = pic2
##    msg.robotid = 1
#    print(msg)
if __name__ == '__main__':
     main()
