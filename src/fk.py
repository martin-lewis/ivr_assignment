#!/usr/bin/env python3

import roslib
import sys
import rospy
from cv2 import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridgeError
from math import sin, pi, sqrt, acos,cos, degrees
from scipy.optimize import least_squares

class forward_kinematics:

    # Defines publisher and subscriber
    def __init__(self):

        # prepare joint variables which will receive data from 
        # the vision components
        self.joint1 = 0 
        self.joint2 = None
        self.joint3 = None 
        self.joint4 = None 

        # initialize the node named
        rospy.init_node('forward_kinematics', anonymous=True)

        #Subscribers for the angles of each joint that we need
        self.joint2_sub = rospy.Subscriber("/robot/joint2_position_controller/command", Float64, self.callback_joint_2)
        self.joint3_sub = rospy.Subscriber("/robot/joint3_position_controller/command", Float64, self.callback_joint_3)
        self.joint4_sub = rospy.Subscriber("/robot/joint4_position_controller/command", Float64, self.callback_joint_4)
        
        # subscribers for end-effector actual position
        self.joint2_sub = rospy.Subscriber("/robot/joint2_position_controller/command", Float64, self.callback_joint_2)
        self.joint3_sub = rospy.Subscriber("/robot/joint3_position_controller/command", Float64, self.callback_joint_3)
        self.joint4_sub = rospy.Subscriber("/robot/joint4_position_controller/command", Float64, self.callback_joint_4)
        
        #Publishers for the calculated joint positions
        self.x_pub = rospy.Publisher("/robot/end_effector_x/command", Float64, queue_size=10)
        self.y_pub = rospy.Publisher("/robot/end_effector_y/command", Float64, queue_size=10)
        self.z_pub = rospy.Publisher("/robot/end_effector_z/command", Float64, queue_size=10)

        print("hello")

    def callback_joint_2(self,msg):
        self.joint2 = msg.data
        self.recompute_end_effector()

    def callback_joint_3(self,msg):
        self.joint3 = msg.data
        self.recompute_end_effector()

    def callback_joint_4(self,msg):
        self.joint4 = msg.data
        self.recompute_end_effector()

    def recompute_end_effector(self):
        if(self.joint1 != None and
            self.joint2 != None and
            self.joint3 != None and
            self.joint4 != None):
            
            print(self.get_forward_kinematics(
                self.joint1,
                self.joint2,
                self.joint3,
                self.joint4))
            # print(degrees(self.joint2),degrees(self.joint3),degrees(self.joint3))

    def get_forward_kinematics(self,q1,q2,q3,q4):
        
        x = -3 * (sin(q1)*sin(q2)*cos(q3) + sin(q3)*cos(q1)) * cos(q4) - 3.5 * sin(q1)*sin(q2)*cos(q3) - 3 * sin(q1)*sin(q4)*cos(q2) - 3.5 * sin(q3)*cos(q1)
        y = 3 * (-sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3)) * cos(q4) - 3.5 * sin(q1)*sin(q3) + 3.5 * sin(q2)*cos(q1)*cos(q3) + 3 * sin(q4)*cos(q1)*cos(q2)
        z = -3 * sin(q2) * sin(q4) + 3 * cos(q2)*cos(q3)*cos(q4) + 3.5 * cos(q2)*cos(q3) + 2.5

        return np.array([x,y,z])

# call the class
def main(args):
  fk = forward_kinematics()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)