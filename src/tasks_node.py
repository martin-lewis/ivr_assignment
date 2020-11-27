#!/usr/bin/env python3

from numpy.lib.function_base import angle
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
from vision import *
from control import *
import message_filters
import asyncio
import os


class task_node:



  def main(self):
    if self.fk == None:
      return

    # self.task_1()
    # self.task2_1()
    self.task_2_2()
    self.task_3_1()
    # self.set_joints(0,0,0)

  def task_1(self):


    #Set the joints according to the sinusodial positions:

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
      fk_observed= self.fk(0,joint_angles[0],joint_angles[1],joint_angles[2])
      fk_real = self.fk(0,target_q2,target_q3 ,target_q4)
      print("\n")

      print("observed     :" + str(redPos))
      # print("non-trian    :" + str(self.detect_in_3D(self.detect_red,self.cv_image2,self.cv_image1)))
      # print("fk observed  :" + str(fk_observed.flatten()) )
      # print("fk real      :" + str(fk_real.flatten()) )
      # print("diff real fk :" + str(np.linalg.norm(fk_real.flatten() - redPos)) )
      print("obs_angs  :" + str(joint_angles))
      print("real_angs :" + str(target_q2) +"," +str(target_q3)+","+ str(target_q4))
      print("\n")
    
  def debug_display(self):
    mx = self.draw_masks(self.cv_image2,"xmask.png")
    # my = self.draw_masks(self.cv_image1,"ymask.png")

    cv2.imshow("yplane/xcam",mx)#cv2.addWeighted(self.cv_image2,0.7,mx,0.3,0))
    # cv2.imshow("xplane/ycam",cv2.addWeighted(self.cv_image1,0.7,my,0.3,0))
    cv2.waitKey(0)

  async def display_img_async(self,wname,img,time):
      cv2.imshow(wname,img)
      cv2.waitKey(time * 1000)
      cv2.destroyAllWindows()

  # comparing FK position to observed position of end effector
  def task2_1(self):

    # we set the arm to 10 angles below and observe the difference between the fk
    # prediction and observed end effector position
    # this function is called every time we receive an image

    # we perform one dummy result at the beginning with the initial joint states being
    # whatever they were before this task was called
    angles = [np.array([0.5,-0.5,0.5,-0.5]),
                      np.array([-0.5,0.5,-0.5,0.5]),
                      np.array([0.5,0.5,0.5,0.5]),
                      np.array([-0.5,-0.5,-0.5,-0.5]),
                      np.array([1,0.5,-1,0.5]),
                      np.array([0.5,1,-1,0.5]),
                      np.array([pi/2,pi/2,0.1,0.1]), # full extension hits the ground
                      np.array([pi,pi/2,0.1,0.1]), # full extension hits the ground
                      np.array([-pi/2,pi/2,1,0.5]),
                      np.array([-pi,pi/2,0.1,0.1])] # full extension hits the ground
    arm_settling_time = 3 # the time we allocate for the arm to align to the values

    filename = os.path.dirname(os.path.realpath(__file__)) + "/../report/task_2_1_results.tex"
    if os.path.exists(filename):
      append_write = 'a' # append if already exists
    else:
      append_write = 'w' # make a new file if not

    f=open(filename,append_write)

    # only act every period

    angle_idx = self.task_context["task2_1-idx"]

    if angle_idx == -1:
      f.write("\\hline \n q1&q2&q3&q4&vision&FK&euclidian distance \\\\ \\hline \n")
      self.task_context["task2_1-start-time"] = rospy.get_time()
      self.set_joints(angles[0][1],
                    angles[0][2],
                    angles[0][3],
                    q1=angles[0][0])

      # update index for next iter
      angle_idx += 1
      self.task_context["task2_1-idx"] = angle_idx

      # dummy phase, don't take a reading, we set the  initial joint values for the first reading
      # we need the arm to settle into position first though

    # we only act in discrete intervals to let the arm settle, we don't need to process every image
    dt =  rospy.get_time() - self.task_context["task2_1-start-time"]
    if dt < arm_settling_time*(angle_idx+1):
      f.close()
      return


    if angle_idx < len(angles): 
      
      curr_arm_angles = angles[angle_idx]

      # observe end effector
      _,_,redPos = self.find_blob_positions()
      # calculate end effector position via fk
      prediction = self.fk(
        curr_arm_angles[0],
        curr_arm_angles[1],
        curr_arm_angles[2],
        curr_arm_angles[3]).flatten()

      # show snapshot at this exact moment of the image used for the measurments
      asyncio.run(self.display_img_async("yplane/xcam-({:.2f},{:.2f},{:.2f},{:.2f})".format(curr_arm_angles[0],curr_arm_angles[1],curr_arm_angles[2],curr_arm_angles[3]),self.cv_image1,arm_settling_time))
      # asyncio.run(self.display_img_async("xplane/ycam({:.2f},{:.2f},{:.2f},{:.2f})".format(curr_arm_angles[0],curr_arm_angles[1],curr_arm_angles[2],curr_arm_angles[3]),self.cv_image2,arm_settling_time//2))
      print(redPos,prediction)

      f.write("{:.2f}&{:.2f}&{:.2f}&{:.2f}&{}&{}&{} \\\\ \\hline \n".format(
        curr_arm_angles[0],
        curr_arm_angles[1],
        curr_arm_angles[2],
        curr_arm_angles[3],
        redPos,
        prediction,
        np.linalg.norm(redPos-prediction)))

      f.close()

      # set the joint values for the next iteration (if there is one ahead)
      if angle_idx < len(angles) - 1:
        self.set_joints(angles[angle_idx+1][1],angles[angle_idx+1][2],angles[angle_idx+1][3],q1=angles[angle_idx+1][0])
      
      # update index for next iter
      self.task_context["task2_1-idx"] = angle_idx+1
    else:
      pass


  def task_2_2(self):

    _,_,redPos = self.find_blob_positions()
    
    if self.first_time:
      self.q_prev_observed = np.array([0,0,0])
      self.set_joints(0,0,0,0)
      time.sleep(1)

      self.first_time = False
      return
    else:
      joint_angles  = self.real_joint_values

    target_end_pos = self.triangulate_in_3D(self.detect_target, self.cv_image2, self.cv_image1) 
    # print(target_end_pos)
    print(target_end_pos)
    q_d = self.closed_control(target_end_pos,
      redPos,
      joint_angles[0],
      joint_angles[1],
      joint_angles[2])

    # publish end effector
    self.endpos_x_pub.publish(redPos[0])
    self.endpos_y_pub.publish(redPos[1])
    self.endpos_z_pub.publish(redPos[2])

    self.q_prev_observed = q_d

    self.set_joints(q_d[0],q_d[1],q_d[2])

  def task_3_1(self):

    _,_,redPos = self.find_blob_positions()
    
    if self.first_time:
      self.q_prev_observed = np.array([0,0,0])
      self.set_joints(0,0,0,0)
      time.sleep(1)

      self.first_time = False
      return
    else:
      joint_angles  = self.real_joint_values

    target_end_pos = self.triangulate_in_3D(self.detect_target, self.cv_image2, self.cv_image1) 
    obstacle_pos = self.triangulate_in_3D(self.detect_box, self.cv_image2, self.cv_image1) 

    q_d = self.closed_control(target_end_pos,
      redPos,
      joint_angles[0],
      joint_angles[1],
      joint_angles[2],
      obstacle_pos)

    # publish end effector
    self.endpos_x_pub.publish(redPos[0])
    self.endpos_y_pub.publish(redPos[1])
    self.endpos_z_pub.publish(redPos[2])

    self.q_prev_observed = q_d

    self.set_joints(q_d[0],q_d[1],q_d[2])
  

  def receive_stereo_image(self,image1,image2):
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(image1, "bgr8")
    except CvBridgeError as e:
      print(e)

    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(image2, "bgr8")
    except CvBridgeError as e:
      print(e)

    self.main()
  
  # Defines publisher and subscriber
  def __init__(self):
    np.set_printoptions(precision=3)

    self.task_context = {
      "task2_1-idx":-1,
      "task2_1-start-time":-1,
      "task2_2-qs":[]
    }
    # prepare subscribers for both the images
    rospy.init_node('task_node', anonymous=True)
    self.image_sub1 = message_filters.Subscriber("/camera1/robot/image_raw",Image)
    self.image_sub2 = message_filters.Subscriber("/camera2/robot/image_raw",Image)

    # synchronise them into one callback
    self.ats = message_filters.ApproximateTimeSynchronizer([self.image_sub1,self.image_sub2],queue_size=1,slop=0.1,allow_headerless=True)
    self.ats.registerCallback(self.receive_stereo_image)

    self.fk,self.jacobian,self.vk,self.green_transform = None,None,None,None
    self.first_time = True 

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

    #Publish the estimated position of the 3 joints
    self.endpos_x_pub = rospy.Publisher("endpos_x", Float64, queue_size=10)
    self.endpos_y_pub = rospy.Publisher("endpos_y", Float64, queue_size=10)
    self.endpos_z_pub = rospy.Publisher("endpos_z", Float64, queue_size=10)

    #Publish the estimated position of the target
    self.est_target_x = rospy.Publisher("observed/target_x", Float64, queue_size=10)
    self.est_target_y = rospy.Publisher("observed/target_y", Float64, queue_size=10)
    self.est_target_z = rospy.Publisher("observed/target_z", Float64, queue_size=10)


    self.q2_sub = rospy.Subscriber("/robot/joint2_position_controller/command", Float64,self.q_x)
    self.q3_sub = rospy.Subscriber("/robot/joint3_position_controller/command", Float64,self.q_y)
    self.q4_sub = rospy.Subscriber("/robot/joint4_position_controller/command", Float64,self.q_z)

    self.real_joint_values = np.array([0.0,0.0,0.0])

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
    self.end_effector_fk_x = rospy.Publisher("fk/end_effector_x", Float64, queue_size=10)
    self.end_effector_fk_y = rospy.Publisher("fk/end_effector_y", Float64, queue_size=10)
    self.end_effector_fk_z = rospy.Publisher("fk/end_effector_z", Float64, queue_size=10)

    self.end_effector_vision_x = rospy.Publisher("vision/end_effector_x", Float64, queue_size=10)
    self.end_effector_vision_y = rospy.Publisher("vision/end_effector_y", Float64, queue_size=10)
    self.end_effector_vision_z = rospy.Publisher("vision/end_effector_z", Float64, queue_size=10)

    # initial time
    self.time_previous_step = np.array([rospy.get_time()])
    # the vector from current to desired position in the last loop
    self.error = np.array([0.0,0.0,0.0])

    # secondary objective function
    self.w_prev = 0
    # joint angles in last loop
    self.q_prev_observed = np.array([0.0,0.0,0.0])
    self.q_prev_cl_output = np.array([0.0,0.0,0.0])
    self.set_joints(0,0,0,0)

    self.fk,self.jacobian,self.vk,self.green_transform = calculate_all()

  def draw_masks(self,img,fname):
    kernel = np.ones((5, 5), np.uint8)
    blank_image = np.zeros(img.shape, np.uint8)

    threshy = cv2.inRange(img, (0,100,100), (10,255,255)) #Thresholds for values
    threshy = cv2.dilate(threshy, kernel, iterations=3)
    comy = self.getCoM(threshy)

    threshb = cv2.inRange(img, (100,0,0), (255,10,10)) #Thresholds for values
    threshb = cv2.dilate(threshb, kernel, iterations=3)
    comb = self.getCoM(threshb)

    threshg = cv2.inRange(img, (0,100,0), (10,255,10)) #Thresholds for values
    threshg = cv2.dilate(threshg, kernel, iterations=3)
    comg = self.getCoM(threshg)

    threshr = cv2.inRange(img, (0,0,100), (10,10,255)) #Thresholds for values
    threshr = cv2.dilate(threshr, kernel, iterations=3)
    comr = self.getCoM(threshr)

    for i in range(len(blank_image)):
      for j in range(len(blank_image[0])):
        if threshy[i,j] != 0:
          blank_image[i,j] = np.array([0,200,200])
        elif threshb[i,j] != 0:
          blank_image[i,j] = np.array([255,0,0])
        elif threshg[i,j] != 0:
          blank_image[i,j] = np.array([0,255,0])
        elif threshr[i,j] != 0: 
          blank_image[i,j] = np.array([0,0,255])

    blank_image = cv2.putText(blank_image, str(comy), (comy[0],comy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) , 2, cv2.LINE_AA)
    blank_image = cv2.putText(blank_image, str(comb), (comb[0],comb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) , 2, cv2.LINE_AA)
    blank_image = cv2.putText(blank_image, str(comg), (comg[0],comg[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) , 2, cv2.LINE_AA)
    blank_image = cv2.putText(blank_image, str(comr), (comr[0],comr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) , 2, cv2.LINE_AA)
    blank_image = cv2.drawMarker(blank_image, (comy[0], comy[1]),(255,255,255), markerType=cv2.MARKER_STAR)
    blank_image = cv2.drawMarker(blank_image, (comb[0], comb[1]),(255,255,255), markerType=cv2.MARKER_STAR)
    blank_image = cv2.drawMarker(blank_image, (comg[0], comg[1]),(255,255,255), markerType=cv2.MARKER_STAR)
    blank_image = cv2.drawMarker(blank_image, (comr[0], comr[1]),(255,255,255), markerType=cv2.MARKER_STAR)

    return blank_image

  def getCoM(self,img):
    M = cv2.moments(img)
    xPos = int(M["m10"] / M["m00"]) #Calculate centre from moments
    yPos = int(M["m01"] / M["m00"])
    return np.array([xPos,yPos])

  #Simple detection method
  def detect_yellow(self, img):
    thresh = cv2.inRange(img, (0,100,100), (10,255,255)) #Thresholds for values
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    return self.getCoM(result) #Positions returned

  def detect_blue(self, img):
    thresh = cv2.inRange(img, (100,0,0), (255,10,10))
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    return self.getCoM(result) #Positions returned


  def detect_green(self, img):
    thresh = cv2.inRange(img, (0,100,0), (10,255,10))
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    return self.getCoM(result) #Positions returned


  def detect_red(self, img):
    thresh = cv2.inRange(img, (0,0,100), (10,10,255))
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(thresh, kernel, iterations=3)
    return self.getCoM(result) #Positions returned


  #Detects the orange sphere that is the target
  def detect_target(self, img):
    template =cv2.imread("/home/martin/catkin_ws/src/ivr_assignment/template-sphere.png", 0) #Loads the template
    thresh = cv2.inRange(img, (0,50,100), (12,75,150)) #Marks all the orange areas out
    if (sum(sum(thresh)) == 0): #If it is obscured
      return None #Return none
    matching = cv2.matchTemplate(thresh, template, 1) #Performs matching between the thresholded data and the template
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching) #Gets the results of the matching
    width, height = template.shape[::-1] #Details of the template to generate the centre
    return np.array([min_loc[0] + width/2, min_loc[1] + height/2]) #Returns the centre of the target

  def detect_box(self, img):
    template =cv2.imread("/home/martin/catkin_ws/src/ivr_assignment/template-box.png", 0) #Loads the template
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
    
    
    screen_pos_y_camera = detect_func(img_xplane)
    screen_pos_x_camera = detect_func(img_yplane)

    if(screen_pos_x_camera is None or screen_pos_y_camera is None):
      return np.array([self.estimateNextPos(detect_func,0),
        self.estimateNextPos(detect_func,1),
        self.estimateNextPos(detect_func,2)])


    ray_x = ray_casting.screen_to_world_ray(self.cam_inv_int_matrix,self.camx_inv_ext_matrix,screen_pos_x_camera)
    ray_y = ray_casting.screen_to_world_ray(self.cam_inv_int_matrix,self.camy_inv_ext_matrix,screen_pos_y_camera)

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
    
    bluePos = self.triangulate_in_3D(self.detect_blue, self.cv_image2, self.cv_image1)
    greenPos = self.triangulate_in_3D(self.detect_green, self.cv_image2, self.cv_image1)
    redPos = self.triangulate_in_3D(self.detect_red, self.cv_image2, self.cv_image1)
    
    return (bluePos,greenPos,redPos)


  def calc_joint_angles(self,bluePos,greenPos,redPos):
    #Joint 1
    blue2green = greenPos - bluePos #Forms vector representing the link
    joint2Angle = clamp(atan2(blue2green[2], blue2green[1]) - pi/2, -pi/2, pi/2) #Angle for joint2
    #Joint 2
    blue2green = rotateX(-joint2Angle, blue2green) #Rotates the vector to remove the affect of joint2
    joint3Angle = clamp(-(atan2(blue2green[2], blue2green[0]) - pi/2),-pi/2,pi/2) #Angle for joint3
    #Joint 3
    green2red = redPos - greenPos #Vector representing link
    projg2rb2g = projection(green2red, blue2green) #Projections

    # i dont think this flipping will ever occur within the limit of this angle
    # so this might be introducing errors with vision going nuts
    # TODO: Check this martin
    # if (euclideanNorm(green2red + projg2rb2g) < euclideanNorm(green2red)):
    #   joint4Angle = pi/2 - angleBetweenVectors(green2red, projg2rb2g) #Flipped if needed
    # else :

    joint4Angle = clamp(angleBetweenVectors(green2red, projg2rb2g),-pi/2,pi/2)
    
    # we need to find out which side of the projg2rb2g the end effector is in 
    # its frame of reference, we use the vector pointing in positive rotation direction of green to red

    # extract z direction vector of the blue frame in the world frame
    z_tip = (self.green_transform(0,joint2Angle) @ np.array([[0],[0],[1],[1]])).flatten()[:-1]
    positive_dir = (z_tip - bluePos)

    # find out direction of green rotation
    if green2red@positive_dir >= 0: # dot product
      joint4Angle *= -1
    
    return np.array([joint2Angle, joint3Angle, joint4Angle]) #Values returned


  # pos_d-    desired position, 
  # pos-      current end effector position
  # q_est-    estimated joint angles
  def closed_control(self,pos_d,pos,q1,q2,q3,obstacle_pos=None):

    #TODO: tweak those 
    q_est = np.array([q1,q2,q3])
    # P gain
    k_p = 1
    k_d = 0.1
    k_0 = 0.0001 
    K_p = np.array([[k_p,0,0],[0,k_p,0],[0,0,k_p]])
    # D gain
    K_d = np.array([[k_d,0,0],[0,k_d,0],[0,0,k_d]])
    damping_factor = 3
    
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
    J[:,0] = 0 # fix joint 0
    J_inv = damped_pseudo_inverse(J,damping_factor)  
    sg_term = 0
    if obstacle_pos is not None:
        # work out the partial derivative of w with respect to q (i.e. q0_d)
      # numerical approximation, i.e. w_now - w_before/ delta_q
      w =  np.linalg.norm(pos - obstacle_pos)#np.sqrt(np.linalg.det(J@J.T))
      delta_w = w - self.w_prev 
      self.w_prev = w

      # the change in angles 
      delta_q = q_est - self.q_prev_cl_output
      self.q_prev_cl_output = q_est
      # q0_d TODO: deal with division by zero
      delta_q[delta_q == 0] = 0.001
      q0_d = (delta_w/(delta_q)) * k_0
      q0_d = np.array([0,q0_d[0],q0_d[1],q0_d[2]])
      sg_term = ((np.eye(4) - J_inv@J) @ q0_d)

    
    # angular velocity of joints  
    errs = (K_p @ np.reshape(self.error,(3,1))) + (K_d @ np.reshape(error_d,(3,1))) 
    dq_d = (J_inv @ errs).flatten() + sg_term
    # new joint angles 
    q_d = np.array([q1,q2,q3]) + (dt * dq_d)[1:]
    q_d = limit_q(q_d)
    return q_d


  def set_joints(self,q2,q3,q4,q1=0):

    if q1 is not None:
      joint1Val = Float64() #Create Float
      joint1Val.data = q1
      self.joint1_pub.publish(joint1Val) #Publish float to joint

    joint2Val = Float64() #Create Float
    joint2Val.data = q2
    self.joint2_pub.publish(joint2Val) #Publish float to joint
    joint3Val = Float64()
    joint3Val.data = q3
    self.joint3_pub.publish(joint3Val)
    joint4Val = Float64()
    joint4Val.data = q4
    self.joint4_pub.publish(joint4Val)
    


  # --- CALLBACKS --- #

  def q_x(self,data):
    self.real_joint_values[0] =float(data.data)

  def q_y(self,data):
    self.real_joint_values[1] = float(data.data)

  def q_z(self,data):
    self.real_joint_values[2] = float(data.data)
    



  # --- UTILS --- #

# call the class
def main(args):
  tn = task_node()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


