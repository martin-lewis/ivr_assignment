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
from kinematics import calculate_all

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):


    # Task 1

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

    #Publish the estimated position of the 3 joints
    self.est_joint2_pub = rospy.Publisher("observed/joint2", Float64, queue_size=10)
    self.est_joint3_pub = rospy.Publisher("observed/joint3", Float64, queue_size=10)
    self.est_joint4_pub = rospy.Publisher("observed/joint4", Float64, queue_size=10)

    # Task 2

    #Publisher end effector positions
    self.end_effector_observed = rospy.Publisher("observed/end_effector", Float64MultiArray, queue_size=0)
    self.end_effector_calculated = rospy.Publisher("calculated/end_effector", Float64MultiArray, queue_size=0)

    self.fk,self.jacobian,self.vk = None,None,None 

    self.fk,self.jacobian,self.vk = calculate_all()

    # initial time
    self.time_previous_step = np.array([rospy.get_time()])
    # the vector from current to desired position in the last loop
    self.error = np.array([0,0,0])

    # secondary objective function
    self.w_prev = 0
    # joint angles in last loop
    self.q_prev_observed = np.array([0,0,0])

    self.q_prev_cl = np.array([0,0,0,0])

  #Simple detection method
  def detect_yellow(self, img):
    thresh = cv2.inRange(img, (0,100,100), (10,145,145)) #Thresholds for values
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
    xPos = int(M["m10"] / M["m00"]) #Calculate centre from moments
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos]) #Positions returned

  def detect_blue(self, img):
    thresh = cv2.inRange(img, (100,0,0), (140,10,10))
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  def detect_green(self, img):
    thresh = cv2.inRange(img, (0,100,0), (10,145,10))
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  def detect_red(self, img):
    thresh = cv2.inRange(img, (0,0,100), (10,10,145))
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
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
    z = (self.detect_in_zaxis(detect_func, img_yplane) + self.detect_in_zaxis(detect_func, img_xplane) ) / 2
    return np.array([x,y,z])


  def find_blob_positions(self):
    bluePos = self.detect_in_3D(self.detect_blue, self.cv_image2, self.cv_image1)
    greenPos = self.detect_in_3D(self.detect_green, self.cv_image2, self.cv_image1)
    redPos = self.detect_in_3D(self.detect_red, self.cv_image2, self.cv_image1)
    
    return (bluePos,greenPos,redPos)

  def calc_joint_angles(self,bluePos,greenPos,redPos):
    #Joint 1
    blue2green = greenPos - bluePos
    normToXZAxis = [0,1,0]
    projGreenXZAxis = self.projectionOntoPlane(blue2green, normToXZAxis)
    joint2Angle = self.angleBetweenVectors(blue2green, projGreenXZAxis)
    if greenPos[1] > 0 :
      joint2Angle = -1 * joint2Angle
    #Joint 2
    normToYZAxis = [1,0,0]
    projGreenYZAxis = self.projectionOntoPlane(blue2green, normToYZAxis)
    joint3Angle = self.angleBetweenVectors(blue2green, projGreenYZAxis)
    if greenPos[0] < 0:
      joint3Angle = -1 * joint3Angle
    #Joint 3
    green2red = redPos - greenPos
    projg2rb2g = self.projection(green2red, blue2green)
    joint4Angle = self.angleBetweenVectors(green2red, projg2rb2g)
    #TODO: Work out which way its turned
    return np.array([joint2Angle, joint3Angle, joint4Angle])


  def euclideanNorm(self, vector):
    total = 0
    for val in vector:
      total = total + pow(val, 2)
    return sqrt(total)

  def angleBetweenVectors(self, v1, v2):
    return acos(np.dot(v1,v2) / (self.euclideanNorm(v1) * self.euclideanNorm(v2)))

  def projectionOntoPlane(self, vector, planeNormal):
    return vector - self.projection(vector, planeNormal)

  def projection(self, v1, v2): #Projects v1 onto v2
    return np.multiply((np.dot(v1,v2) / pow(self.euclideanNorm(v2),2)), v2)

  def publish_forward_kinematics_results(self,q1,q2,q3,q4,observedRedBlobPosition):
    prediction = self.fk(q1,q2,q3,q4)
    real = observedRedBlobPosition

    msg_calculated = Float64MultiArray()
    msg_calculated.data = prediction

    msg_observed = Float64MultiArray()
    msg_observed.data = real
    self.end_effector_calculated.publish(msg_calculated)
    self.end_effector_observed.publish(msg_observed)


  # pos_d-    desired position, 
  # pos-      current end effector position
  # q_est-    estimated joint angles
  def closed_control(self,pos_d,pos,q_est,secondary_objective_function):

    #TODO: tweak those 
    # P gain
    k_p = 1
    k_d = 0.01
    k_0 = 0
    K_p = np.array([[k_p,0,0],[0,k_p,0],[0,0,k_p]])
    # D gain
    K_d = np.array([[k_d,0,0],[0,k_d,0],[0,0,k_d]])

    # loop time
    cur_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time

    # partial derivative of (pos_d - pos) with respect to time
    # (numerical approximation) 
    error_d = ((pos_d - pos) - self.error)/dt
    # estimate error
    self.error = pos_d-pos
    # pseudo inverse
    J = self.jacobian(q_est[0],q_est[1],q_est[2],q_est[3])
    J_inv = np.linalg.pinv(J)  

      # work out the partial derivative of w with respect to q (i.e. q0_d)
    # numerical approximation, i.e. w_now - w_before/ delta_q
    w = secondary_objective_function(q_est)
    delta_w = w - self.w_prev 
    self.w_prev = w

    # the change in angles 
    delta_q = q_est - self.q_prev_cl
    self.q_prev_cl = q_est

    # q0_d TODO: deal with division by zero
    q0_d = delta_w/(delta_q + 0.000001) * k_0

    # angular velocity of joints  
    errs = (K_p @ np.reshape(self.error,(3,1))) + (K_d @ np.reshape(error_d,(3,1))) 
    dq_d = (J_inv @ errs).flatten() + ((np.eye(4) - J_inv@J) @ q0_d)
    # new joint angles 
    q_d = q_est + (dt * dq_d)*0.001
 
    return q_d

  def closed_control_w_secondary_task(self,pos_d,pos,q_est,secondary_objective_function):
    pass
    # #TODO: tweak those 
    # # P gain
    # K_p = np.array([[10,0,0],[0,10,0],[0,0,10]])
    # # D gain
    # K_d = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
    # # step rate
    # k_0 = 0.1

    # # loop time
    # cur_time = np.array([rospy.get_time()])
    # dt = cur_time - self.time_previous_step
    # self.time_previous_step = cur_time

    # # partial derivative of (pos_d - pos) with respect to time
    # # (numerical approximation) 
    # error_d = ((pos_d - pos) - self.error)/dt
    # # estimate error
    # self.error = pos_d-pos
    # # pseudo inverse
    
    # J = self.jacobian(q_est[0],q_est[1],q_est[2],q_est[3])
    # J_inv = np.linalg.pinv(J)  

    # # work out the partial derivative of w with respect to q (i.e. q0_d)
    # # numerical approximation, i.e. w_now - w_before/ delta_q
    # w = secondary_objective_function(q_est)

    # delta_w = w - self.w_prev 
    # self.w_prev = w

    # # the change in angles 
    # delta_q = q_est - self.q_prev_cl
    # self.q_prev_cl = q_est

    # # q0_d TODO: deal with division by zero
    # q0_d = delta_w/(delta_q+0.0000001) * k_0

    # # angular velocity of joints  
    # secondary_goal_term = (np.eye(4) - J_inv@J) @ q0_d
    # error_p_gain = np.dot(K_p,self.error.transpose())
    # error_d_gain = np.dot(K_d,error_d.transpose())
    # error_sum = error_d_gain + error_p_gain 
    # dq_d =np.dot(J_inv,(error_sum))+ secondary_goal_term
    # # new joint angles 
    # q_d = q_est + (dt * dq_d)  
    # return q_d

  def set_joints(self,q2,q3,q4):
    joint2Val = Float64() #Create Float
    joint2Val.data = q2
    self.joint2_pub.publish(joint2Val) #Publish float to joint
    joint3Val = Float64()
    joint3Val.data = q3
    self.joint3_pub.publish(joint3Val)
    joint4Val = Float64()
    joint4Val.data = q4
    self.joint4_pub.publish(joint4Val)

  def task_1(self,data):

    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)


    #Set the joints according to the sinusodial positions
    target_q2 = (pi/2) * sin((pi/15) * rospy.get_time())
    target_q3 = (pi/2) * sin((pi/18) * rospy.get_time())
    target_q4 = (pi/2) * sin((pi/20) * rospy.get_time())

    self.set_joints(target_q2,
                    target_q3,
                    target_q4)


    bluePos,greenPos,redPos = self.find_blob_positions()
    joint_angles = (self.calc_joint_angles(bluePos,greenPos,redPos))


    #Publising estimated joint angles
    self.est_joint2_pub.publish(joint_angles[0])
    self.est_joint3_pub.publish(joint_angles[1])
    self.est_joint4_pub.publish(joint_angles[2])

    # for debugging


    if self.fk is not None:
      fk_observed= self.fk(0,joint_angles[0],joint_angles[1],joint_angles[2])
      fk_real = self.fk(0,target_q2,target_q3 ,target_q4)
      print("observed      :" + str(redPos))
      print("fk observed   :" + str(fk_observed.flatten()) )
      print("fk real       :" + str(fk_real.flatten()) )
      print("diff real fk  :" + str(np.linalg.norm(fk_real.flatten() - redPos)) )

      print("\n")

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)


  def task_2(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    try:
      bluePos,greenPos,redPos = self.find_blob_positions()
      joint_angles = (self.calc_joint_angles(bluePos,greenPos,redPos))
    except:
      joint_angles = self.q_prev
      redPos = self.fk(0,joint_angles[0],joint_angles[1],joint_angles[2]).flatten()

    joint_angles = ((joint_angles - self.q_prev_observed) + self.q_prev_observed) 
    self.q_prev_observed = joint_angles
    
    # target_pos = self.detect_in_3D(self.detect_target, self.cv_image2, self.cv_image1)
    target_q2 = (pi/2) * sin((pi/15) * rospy.get_time())
    target_q3 = (pi/2) * sin((pi/18) * rospy.get_time())
    target_q4 = (pi/2) * sin((pi/20) * rospy.get_time())

    target_end_pos = np.array([target_q2,target_q3,target_q4])

    print(target_end_pos)

    q_d = self.closed_control(target_end_pos,redPos,np.array([0,joint_angles[0],joint_angles[1],joint_angles[2]]),
      lambda x: np.sqrt(np.linalg.det(self.jacobian(0,x[0],x[1],x[3]) @ self.jacobian(0,x[0],x[1],x[3]).T)))

    self.set_joints(q_d[1],q_d[2],q_d[3])
    # attempt at smoothing noise
    # q_next = (self.q_prev[1:4] -  q_d[1:4]) * 0.1 + self.q_prev[1:4]
    # self.set_joints(q_next[0],q_next[1],q_next[2])
    # self.q_prev = np.array([0,q_next[0],q_next[1],q_next[2]])


  

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)
    
  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):

    # self.task_1(data)
    self.task_2(data)
    # self.set_joints(0,0,0)


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


