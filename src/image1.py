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
from math import sin, pi, sqrt, acos
from scipy.optimize import least_squares

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

  #Detects the orange sphere that is the target
  def detect_target(self, img):
    template =cv2.imread("~/catkin_ws/src/ivr_assignment/template-sphere.png", 0) #Loads the template
    thresh = cv2.inRange(img, (0,50,100), (12,75,150)) #Marks all the orange areas out
    matching = cv2.matchTemplate(thresh, template, 1) #Performs matching between the thresholded data and the template
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching) #Gets the results of the matching
    width, height = template.shape[::-1] #Details of the template to generate the centre
    return np.array([min_loc[0] + width/2, min_loc[1] + height/2]) #Returns the centre of the target

  def detect_box(self, img):
    template =cv2.imread("~/catkin_ws/src/ivr_assignment/template-box.png", 0) #Loads the template
    thresh = cv2.inRange(img, (0,50,100), (12,75,150)) #Marks all the orange areas out
    matching = cv2.matchTemplate(thresh, template, 1) #Performs matching between the thresholded data and the template
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching) #Gets the results of the matching
    width, height = template.shape[::-1] #Details of the template to generate the centre
    return np.array([min_loc[0] + width/2, min_loc[1] + height/2]) #Returns the centre of the target

  #Calculates the pixel to 2 ratio using the 2.5m distance between the blue and yellow joints
  def pixel2metre(self, img):
    yellow = self.detect_yellow(img)
    blue = self.detect_blue(img)
    pixelDist = sqrt((yellow[0] - blue[0])**2 + (yellow[1] - blue[1]) ** 2) #Euclidean Distance
    return 2.5 / pixelDist

  #Detects the position (w.r.t. yellow joint in metres) of an object on the y axis given a function that returns its position and the view of the y-plane
  def detect_in_yaxis(self, detect_func, img_yplane):
    distance = detect_func(img_yplane) - self.detect_yellow(img_yplane)
    return distance[0] * self.pixel2metre(img_yplane)

  #Detects the position (w.r.t. yellow joint in metres) of an object on the z axis given a function that returns its position and the view of the y-plane
  def detect_in_zaxis(self, detect_func, img_yplane):
    distance = self.detect_yellow(img_yplane) - detect_func(img_yplane)
    return distance[1] * self.pixel2metre(img_yplane)
  
  #Detects the position (w.r.t. yellow joint in metres) of an object on the x axis given a function that returns its position and the view of the x-plane
  def detect_in_xaxis(self, detect_func, img_xplane):
    distance = detect_func(img_xplane) - self.detect_yellow(img_xplane)
    return distance[0] * self.pixel2metre(img_xplane)

  #Detects in 3D the position of an object w.r.t. the yellow joint
  def detect_in_3D(self, detect_func, img_xplane, img_yplane):
    x = self.detect_in_xaxis(detect_func, img_xplane)
    y = self.detect_in_yaxis(detect_func, img_yplane)
    z = self.detect_in_zaxis(detect_func, img_yplane)
    return np.array([x,y,z])
  
  def F1(self, x):
    matrix = np.array([[np.cos(x[1]), 0, np.sin(x[1])], [np.sin(x[0])*np.sin(x[1]), np.cos(x[0]), -1*np.sin(x[0]) * np.cos(x[1])],
    [-1*np.sin(x[1])*np.cos(x[0]), np.sin(x[0]) , np.cos(x[0])*np.cos(x[1])]])
    blue = self.detect_in_3D(self.detect_blue, self.cv_image2, self.cv_image1)
    green = self.detect_in_3D(self.detect_green, self.cv_image2, self.cv_image1)
    green_initial = blue
    green_initial[2] = green_initial[2] + 3.5
    sumof = np.dot(matrix @ green_initial, green)
    print(sumof)
    return sumof
  
  def calc_joint_angles(self):
    #bluePos = self.detect_in_3D(self.detect_blue, self.cv_image2, self.cv_image1)
    greenPos = self.detect_in_3D(self.detect_green, self.cv_image2, self.cv_image1)
    normToXAxis = [0,1,0]
    projGreenXAxis = greenPos - ( np.multiply((np.dot(greenPos, normToXAxis) / pow(self.euclideanNorm(normToXAxis), 2)), normToXAxis ))
    #print(greenPos)
    #print(projGreenXAxis)
    angle = self.angleBetweenVectors(greenPos, projGreenXAxis)
    if greenPos[1] < 0 :
      return angle
    else:
      return -1 * angle

  def euclideanNorm(self, vector):
    total = 0
    for val in vector:
      total = total + pow(val, 2)
    return sqrt(total)

  def angleBetweenVectors(self, v1, v2):
    return acos(np.dot(v1,v2) / (self.euclideanNorm(v1) * self.euclideanNorm(v2)))

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

    vals = (self.calc_joint_angles())
    print(joint2Val.data)
    print(joint3Val.data)
    print(vals)
    #print("Diffs:")
    #print(abs(joint2Val.data - self.calc_joint_angles()))
    #print(abs(joint3Val.data - self.calc_joint_angles()[1]))

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


