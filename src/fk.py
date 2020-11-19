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

def forward_kinematics(q1,q2,q3,q4):
    
    x = 3 * (sin(q1)*sin(q2)*cos(q3) + sin(q3)*cos(q1)) * cos(q4) + 3.5 * sin(q1)*sin(q2)*cos(q3) + 3 * sin(q1)*sin(q4)*cos(q2) + 3.5 * sin(q3)*cos(q1)
    y = 3 * (-sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3)) * cos(q4) + 3.5 * sin(q1)*sin(q3) - 3.5 * sin(q2)*cos(q1)*cos(q3) - 3 * sin(q4)*cos(q1)*cos(q2)
    z = -3 * sin(q2) * sin(q4) + 3 * cos(q2)*cos(q3)*cos(q4) + 3.5 * cos(q2)*cos(q3) + 2.5

    return np.array([x,y,z])
