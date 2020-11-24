import roslib
import sys
import rospy
from cv2 import cv2
import numpy as np
from std_msgs.msg import String
from kinematics import calculate_all
import math

# retrieves projection matrix for each camera
def get_projection_matrix(camera_axis="x"):
    
    if (camera_axis != "x") and (camera_axis != "y"):
        print("invalid camera axis")
        raise Exception

    camera_x = camera_axis == "x"
    hfov = 1.3962634

    focal_length = (800/2) / math.tan(hfov/2)
    optical_centre = np.array([0.5,0.5])
    return np.array([[focal_length,    0,              optical_centre[0]],
                    [0,               focal_length,   optical_centre[0]],
                    [0,               0,              1]])


print(get_projection_matrix())