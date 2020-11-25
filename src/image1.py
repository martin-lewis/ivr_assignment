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
from math import sin, pi, sqrt, acos, atan2
from kinematics import calculate_all
import ray_casting
import time 

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):

    self.fk,self.jacobian,self.vk = None,None,None 
    self.first_time = True 
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.image_sub1 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    #Publisher each of the 4 joints
    self.joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

    #Publish the estimated position of the 3 joints
    self.est_joint2_pub = rospy.Publisher("observed/joint2", Float64, queue_size=10)
    self.est_joint3_pub = rospy.Publisher("observed/joint3", Float64, queue_size=10)
    self.est_joint4_pub = rospy.Publisher("observed/joint4", Float64, queue_size=10)

    #Publish the estimated position of the target
    self.est_target_x = rospy.Publisher("observed/target_x", Float64, queue_size=10)
    self.est_target_y = rospy.Publisher("observed/target_y", Float64, queue_size=10)
    self.est_target_z = rospy.Publisher("observed/target_z", Float64, queue_size=10)

    #real target coordinates
    self.t1_sub = rospy.Subscriber("/target/x_position_controller/command", Float64,self.o_x)
    self.t2_sub = rospy.Subscriber("/target/y_position_controller/command", Float64,self.o_y)
    self.t3_sub = rospy.Subscriber("/target/z_position_controller/command", Float64,self.o_z)
    self.o1_sub = rospy.Subscriber("/target2/x2_position_controller/command", Float64,self.t_x)
    self.o2_sub = rospy.Subscriber("/target2/y2_position_controller/command", Float64,self.t_y)
    self.o3_sub = rospy.Subscriber("/target2/z2_position_controller/command", Float64,self.t_z)
    self.obstacle_pos = np.array([0.0,0.0,0.0])
    self.target_pos = np.array([0.0,0.0,0.0])

    #Dictionary Holds the last 5 positions of each item
    self.prevPos = {
      self.detect_yellow : [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])],
      self.detect_blue : [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])],
      self.detect_green : [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])],
      self.detect_red : [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])],
      self.detect_box : [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])],
      self.detect_target : [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])],
    }

    # ray casting setup
    self.cam_inv_int_matrix = np.linalg.inv(ray_casting.get_projection_matrix())
    self.camx_inv_ext_matrix = ray_casting.get_camera_to_world_origin_matrix(camera_axis="x")
    self.camy_inv_ext_matrix = ray_casting.get_camera_to_world_origin_matrix(camera_axis="y")
    self.camx_full_matrix = ray_casting.get_camera_matrix(camera_axis="x")
    self.camy_full_matrix = ray_casting.get_camera_matrix(camera_axis="y")

    #Publisher end effector positions
    self.end_effector_observed = rospy.Publisher("observed/end_effector", Float64MultiArray, queue_size=0)
    self.end_effector_calculated = rospy.Publisher("calculated/end_effector", Float64MultiArray, queue_size=0)
    # initial time
    self.time_previous_step = np.array([rospy.get_time()])
    # the vector from current to desired position in the last loop
    self.error = np.array([0.0,0.0,0.0])

    # secondary objective function
    self.w_prev = 0
    # joint angles in last loop
    self.q_prev_observed = np.array([0.0,0.0,0.0])

    self.q_prev_cl = np.array([0.0,0.0,0.0])


    self.fk,self.jacobian,self.vk = calculate_all()


  def o_x(self,data):
    self.obstacle_pos[0] = float(data.data)

  def o_y(self,data):
    self.obstacle_pos[1] = float(data.data)

  def o_z(self,data):
    self.obstacle_pos[2] = float(data.data) - 1.25

  def t_x(self,data):
    self.target_pos[0] =float(data.data)

  def t_y(self,data):
    self.target_pos[1] = float(data.data)

  def t_z(self,data):
    self.target_pos[2] = float(data.data)- 1.25
    
  #Simple detection method
  def detect_yellow(self, img):
    thresh = cv2.inRange(img, (0,100,100), (10,255,255)) #Thresholds for values
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
    xPos = int(M["m10"] / M["m00"]) #Calculate centre from moments
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos]) #Positions returned

  def detect_blue(self, img):
    thresh = cv2.inRange(img, (100,0,0), (255,10,10))
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  def detect_green(self, img):
    thresh = cv2.inRange(img, (0,100,0), (10,255,10))
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  def detect_red(self, img):
    thresh = cv2.inRange(img, (0,0,100), (10,10,255))
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    M = cv2.moments(result)
    xPos = int(M["m10"] / M["m00"])
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos, yPos])

  #Detects the orange sphere that is the target
  def detect_target(self, img):
    template =cv2.imread("/home/maks/catkin_ws/src/ivr_assignment/template-sphere.png", 0) #Loads the template
    thresh = cv2.inRange(img, (0,50,100), (12,75,150)) #Marks all the orange areas out
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    matching = cv2.matchTemplate(thresh, template, 1) #Performs matching between the thresholded data and the template
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching) #Gets the results of the matching
    width, height = template.shape[::-1] #Details of the template to generate the centre
    return np.array([min_loc[0] + width/2, min_loc[1] + height/2]) #Returns the centre of the target

  def detect_box(self, img):
    template =cv2.imread("/home/maks/catkin_ws/src/ivr_assignment/template-box.png", 0) #Loads the template
    thresh = cv2.inRange(img, (0,50,100), (12,75,150)) #Marks all the orange areas out
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
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
    try :
      distance = detect_func(img_yplane) - self.detect_yellow(img_yplane)
      return distance[0] * self.pixel2metre(img_yplane)
    except:
      return None

  #Detects the position (w.r.t. yellow joint in metres) of an object on the z axis given a function that returns its position and the view of the y-plane
  def detect_in_zaxis(self, detect_func, img_yplane):
    try:
      distance = self.detect_yellow(img_yplane) - detect_func(img_yplane)
      return distance[1] * self.pixel2metre(img_yplane)
    except:
      return None
  
  #Detects the position (w.r.t. yellow joint in metres) of an object on the x axis given a function that returns its position and the view of the x-plane
  def detect_in_xaxis(self, detect_func, img_xplane):
    try:
      distance = detect_func(img_xplane) - self.detect_yellow(img_xplane)
      return distance[0] * self.pixel2metre(img_xplane)
    except:
      return None

  #Detects in 3D the position of an object w.r.t. the yellow joint
  def detect_in_3D(self, detect_func, img_xplane, img_yplane):
    x = self.detect_in_xaxis(detect_func, img_xplane)
    if (x == None):
      print("X Fail")
      x = self.estimateNextPos(detect_func, 0)
    y = self.detect_in_yaxis(detect_func, img_yplane)
    if (y == None):
      print("Y Fail")
      y = self.estimateNextPos(detect_func, 1)
    try:
      z = (self.detect_in_zaxis(detect_func, img_yplane) + self.detect_in_zaxis(detect_func, img_xplane) ) / 2
    except:
      print("Z fail")
      z = self.estimateNextPos(detect_func, 2)

    #Adds this value to the dict containing the previous values
    if (len(self.prevPos[detect_func]) < 5):
      self.prevPos[detect_func].append(np.array([x,y,z]))
    else:
      del (self.prevPos[detect_func])[0]
      self.prevPos[detect_func].append(np.array([x,y,z]))

    return np.array([x,y,z])
  

  #WARNING: TODO: make sure yellow blob is origin (pretty sure it works)
  def triangulate_in_3D(self,detect_func, img_xplane, img_yplane):
    
    
    screen_pos_x = detect_func(img_xplane)
    screen_pos_y = detect_func(img_yplane)

    if(screen_pos_x is None):
      screen_pos_x = self.estimateNextPos(detect_func,img_xplane)

    if(screen_pos_y is None):
      screen_pos_y = self.estimateNextPos(detect_func,img_yplane)

    ray_x = ray_casting.screen_to_world_ray(self.cam_inv_int_matrix,self.camx_inv_ext_matrix,screen_pos_x)
    ray_y = ray_casting.screen_to_world_ray(self.cam_inv_int_matrix,self.camy_inv_ext_matrix,screen_pos_y)

    approx_intersect = ray_casting.ray_intersect(ray_x,ray_y)

    return approx_intersect

  def estimateNextPos(self, detect_func, axis): #Note for axis 0 = x 1 = y 2 = z
    previous = self.prevPos[detect_func] #Gets the set of previous coordinates
    previousVals = [] #List to hold the ones we want
    for coords in previous:
      previousVals.append(coords[axis]) #Gets the value for the axis we want
    aveChange = 0
    for i in range(len(previousVals) - 2):
      aveChange = aveChange + previousVals[i] - previousVals[i+1] #Works out the change between each pair
    ave = aveChange / (len(previousVals) - 2) # find the average change
    #print(len(previousVals))
    return previousVals[len(previousVals) - 1] + ave #Add the average change to the last value

  def find_blob_positions(self):
    
    """
    bluePos = self.detect_in_3D(self.detect_blue, self.cv_image2, self.cv_image1)
    greenPos = self.detect_in_3D(self.detect_green, self.cv_image2, self.cv_image1)
    redPos = self.detect_in_3D(self.detect_red, self.cv_image2, self.cv_image1)
    """
    bluePos = self.triangulate_in_3D(self.detect_blue, self.cv_image2, self.cv_image1)
    greenPos = self.triangulate_in_3D(self.detect_green, self.cv_image2, self.cv_image1)
    redPos = self.triangulate_in_3D(self.detect_red, self.cv_image2, self.cv_image1)
    
    return (bluePos,greenPos,redPos)

  def calc_joint_angles(self,bluePos,greenPos,redPos):
    #Joint 1
    blue2green = greenPos - bluePos
    '''
    normToXZAxis = [0,1,0]
    projGreenXZAxis = self.projectionOntoPlane(blue2green, normToXZAxis)
    joint2Angle = self.angleBetweenVectors(blue2green, projGreenXZAxis)
    if greenPos[1] > 0 :
      joint2Angle = -1 * joint2Angle
    '''
    joint2Angle = atan2(blue2green[2], blue2green[1]) - pi/2
    #Joint 2
    '''
    normToYZAxis = [1,0,0]
    projGreenYZAxis = self.projectionOntoPlane(blue2green, normToYZAxis)
    joint3Angle = self.angleBetweenVectors(blue2green, projGreenYZAxis)
    if greenPos[0] < 0:
      joint3Angle = -1 * joint3Angle
    '''
    blue2green = self.rotateX(-joint2Angle, blue2green)
    joint3Angle = atan2(blue2green[2], blue2green[0]) - pi/2
    #Joint 3
    green2red = redPos - greenPos
    projg2rb2g = self.projection(green2red, blue2green)
    if (self.euclideanNorm(green2red + projg2rb2g) < self.euclideanNorm(green2red)):
      joint4Angle = pi /2 - self.angleBetweenVectors(green2red, projg2rb2g)
    else :
      joint4Angle = self.angleBetweenVectors(green2red, projg2rb2g)

    #TODO: Work out which way its turned
    return np.array([joint2Angle, joint3Angle, joint4Angle])

  def rotateX(self, theta, coords):
    matrix = np.array([[1, 0, 0], [0, np.cos(theta), -1 * np.cos(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(matrix, coords)

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
  def closed_control(self,pos_d,pos,q1,q2,q3,obstacle_pos=None):

    #TODO: tweak those 
    q_est = np.array([q1,q2,q3])
    # P gain
    k_p = 0.3 #0.3
    k_d = 0.3 #0.2
    k_0 = 0 #0.1
    K_p = np.array([[k_p,0,0],[0,k_p,0],[0,0,k_p]])
    # D gain
    K_d = np.array([[k_d,0,0],[0,k_d,0],[0,0,k_d]])


    if self.first_time:
      self.error = np.array([0.0,0.0,0.0])
      self.time_previous_step = np.array([rospy.get_time()])
      return q_est
      
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
    J = self.jacobian(0,q1,q2,q3)
    J_inv = np.linalg.pinv(J)  

    sg_term = 0
    if obstacle_pos is not None:
        # work out the partial derivative of w with respect to q (i.e. q0_d)
      # numerical approximation, i.e. w_now - w_before/ delta_q
      w = np.linalg.norm(pos - obstacle_pos)
      delta_w = w - self.w_prev 
      self.w_prev = w

      # the change in angles 
      delta_q = q_est - self.q_prev_cl
      self.q_prev_cl = q_est

      # q0_d TODO: deal with division by zero
      q0_d = delta_w/(delta_q + 0.000001) * k_0
      q0_d = np.array([0,q0_d[0],q0_d[1],q0_d[2]])
      sg_term = ((np.eye(4) - J_inv@J) @ q0_d)

    # angular velocity of joints  
    errs = (K_p @ np.reshape(self.error,(3,1))) + (K_d @ np.reshape(error_d,(3,1))) 
    dq_d = (J_inv @ errs).flatten() + sg_term
    # new joint angles 
    q_d = np.array([q1,q2,q3]) + (dt * dq_d)[1:]

    # q_d = self.limit_q(q_d)
  
    return q_d
  def limit_q(self,q):
    return np.array([x for x in q])
  
  def limit_angle(self,a):
    if a > pi/2 or a < -pi/2:
      return min(max(a,pi/2),-pi/2) 

  def set_joints(self,q2,q3,q4,q0=None):
    joint2Val = Float64() #Create Float
    joint2Val.data = q2
    self.joint2_pub.publish(joint2Val) #Publish float to joint
    joint3Val = Float64()
    joint3Val.data = q3
    self.joint3_pub.publish(joint3Val)
    joint4Val = Float64()
    joint4Val.data = q4
    self.joint4_pub.publish(joint4Val)
    
    if q0 is not None:
      joint1Val = Float64() #Create Float
      joint1Val.data = q2
      self.joint1_pub.publish(joint1Val) #Publish float to joint

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

    #Target Stuff
    targetPos = self.triangulate_in_3D(self.detect_target, self.cv_image2, self.cv_image1)
    self.est_target_x.publish(targetPos[0])
    self.est_target_y.publish(targetPos[1])
    self.est_target_z.publish(targetPos[2])
    # print("observed ang :" +str(joint_angles))
    # for debugging
    if self.fk is not None:
      fk_observed= self.fk(0,joint_angles[0],-joint_angles[1],-joint_angles[2])
      fk_real = self.fk(0,target_q2,target_q3 ,target_q4)
      print("\n")

      print("observed     :" + str(redPos))
      print("non-trian    :" + str(self.detect_in_3D(self.detect_red,self.cv_image2,self.cv_image1)))
      # print("fk observed  :" + str(fk_observed.flatten()) )
      print("fk real      :" + str(fk_real.flatten()) )
      # print("diff real fk :" + str(np.linalg.norm(fk_real.flatten() - redPos)) )
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

    bluePos,greenPos,redPos = self.find_blob_positions()

    if self.first_time:
      joint_angles = np.array([0.0,0.0,0.0])
      self.set_joints(joint_angles[0],joint_angles[1],joint_angles[2])
      self.first_time = False
      
      time.sleep(2)
    else:
      joint_angles = self.q_prev_observed#self.calc_joint_angles(bluePos,greenPos,redPos)
    
    target_end_pos = self.target_pos#self.triangulate_in_3D(self.detect_target, self.cv_image2, self.cv_image1)
    obstacle_pos =self.obstacle_pos #self.triangulate_in_3D(self.detect_box, self.cv_image2, self.cv_image1)
    #WARNING: DO NOT USE ANGLE 0
    q_d = self.closed_control(obstacle_pos,
      redPos,
      joint_angles[0],
      joint_angles[1],
      joint_angles[2],
      obstacle_pos)
    
    self.q_prev_observed = q_d
    self.set_joints(q_d[0],q_d[1],q_d[2])
    
    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)
    
  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # print(self.fk)
    if self.fk == None:
      return

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


