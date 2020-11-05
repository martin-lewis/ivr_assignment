#!/usr/bin/env python3

import roslib
import sys
import rospy
from cv2 import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from math import sin, pi, sqrt


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.image_sub1 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    #Publisher each of the 3 joints
    self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

  #Simple detection method
  def detect_yellow(self, img):
    thresh = cv2.inRange(img, (0,100,100), (10,145,145)) #Thresholds for values
    M = cv2.moments(thresh)
    xPos = int(M["m10"] / M["m00"]) #Calculate centre from moments
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos]) #Positions returned

  def detect_blue(self, img):
    thresh = cv2.inRange(img, (100,0,0), (140,10,10))
    M = cv2.moments(thresh)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  def detect_green(self, img):
    thresh = cv2.inRange(img, (0,100,0), (10,145,10))
    M = cv2.moments(thresh)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  def detect_red(self, img):
    thresh = cv2.inRange(img, (0,0,100), (10,10,145))
    M = cv2.moments(thresh)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  #Calculates the pixel to 2 ratio using the 2.5m distance between the blue and yellow joints
  def pixel2metre(self, img):
    yellow = self.detect_yellow(img)
    blue = self.detect_blue(img)
    pixelDist = sqrt((yellow[0] - blue[0])**2 + (yellow[1] - blue[1]) ** 2) #Euclidean Distance
    return 2.5 / pixelDist

  def green_in_yaxis(self, img_yplane):
    distance = self.detect_green(img_yplane) - self.detect_yellow(img_yplane)
    return distance[0] * self.pixel2metre(img_yplane)
    
  def green_in_zaxis(self, img_yplane):
    distance = self.detect_yellow(img_yplane) - self.detect_green(img_yplane)
    return distance[1] * self.pixel2metre(img_yplane)

  def green_in_xaxis(self, img_xplane):
    distance = self.detect_green(img_xplane) - self.detect_yellow(img_xplane) 
    return distance[0] * self.pixel2metre(img_xplane)

  def green_in_3D(self, img_yplane, img_xplane):
    x = self.green_in_xaxis(img_xplane)
    y = self.green_in_yaxis(img_yplane)
    z = self.green_in_zaxis(img_yplane)
    return np.array([x,y,z])

  def calc_joint_angles(self, img):
    blue = self.detect_blue(img)
    green = self.detect_green(img)
    red = self.detect_red(img)

    joint2 = np.arctan2(blue[0] - green[0], blue[1] - green[1])

    return joint2

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    #Set the joints according to the sinusodial positions
    joint2Val = Float64() #Create Float
    joint2Val.data = (pi/2) * sin((pi/15) * rospy.get_time()) #Set floats values
    self.joint2_pub.publish(joint2Val) #Publish float to joint
    joint3Val = Float64()
    joint3Val.data = (pi/2) * sin((pi/18) * rospy.get_time())
    self.joint3_pub.publish(joint3Val)
    joint4Val = Float64()
    joint4Val.data = (pi/2) * sin((pi/20) * rospy.get_time())
    self.joint4_pub.publish(joint4Val)

    #print(joint2Val.data)
    #print(self.calc_joint_angles(self.cv_image1))
    #print("Diff")
    #print(abs(joint2Val.data - self.calc_joint_angles(self.cv_image1)))
    #print(self.detect_yellow(self.cv_image1))

    print(self.green_in_3D(self.cv_image1, self.cv_image2))

    im1=cv2.imshow('window1', self.cv_image1)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)
    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)
  
  #Additional Callback used to get the image from the other cameras (camera2)
  def callback2(self,data):
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8") #New image stored under self.cv_image2
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


