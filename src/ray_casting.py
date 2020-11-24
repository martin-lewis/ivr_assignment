import roslib
import sys
import rospy
from cv2 import cv2
import numpy as np
from std_msgs.msg import String
from kinematics import calculate_all
import math
from math import cos,sin

hfov = 1.3962634
focal_length = (800/(math.tan(hfov/2)))/2 


def get_projection_matrix():

    
    optical_centre = np.array([400,400])
    return np.array([[focal_length,    0,              optical_centre[0]],
                    [0,               focal_length,   optical_centre[1]],
                    [0,               0,              1]])


def get_world_to_camera_origin_matrix(camera_axis="x"):
    if (camera_axis != "x") and (camera_axis != "y"):
        print("invalid camera axis")
        raise Exception

    a = 0
    b = 0
    c = 0

    if camera_axis == "x":
        a = -math.pi/2
        b = math.pi/2
        c = 0
    else:
        pass
    return np.array([[cos(a)*cos(b),    cos(a)*sin(b)*cos(c) - sin(a)*cos(c),   cos(a)*sin(b)*cos(c) + sin(a)*sin(c),   0],
                    [sin(a)*cos(b),     sin(a)*sin(b)*sin(c) + cos(a)*cos(c),   sin(a)*sin(b)*cos(c) - cos(a)*cos(c),   6],
                    [-sin(b),           cos(b)*sin(c),                          cos(b)*cos(c),                          16]
                    ])




# the height of the maximum rectangle seen by camera at distance of 1m
fh = 2.0 * math.tan(hfov * 0.5)

# fh=5.67*2
print(fh/2)
# corners of the image (with the coordinates Y facing down, and Z away from the camera)

# center
# print(get_projection_matrix()@np.array([[0],[0],[1],[1]]))

# # top left (image 0,0 )
# print(get_projection_matrix()@np.array([[-fh/2],[-fh/2],[1],[1]]))
# # top right 
# print(get_projection_matrix()@np.array([[fh/2],[-fh/2],[1],[1]]))
# # bottom right 
# print(get_projection_matrix()@np.array([[fh/2],[fh/2],[1],[1]]))
# # bottom left
# print(get_projection_matrix()@np.array([[-fh/2],[fh/2],[1],[1]]))

a = (get_world_to_camera_origin_matrix("x")@np.array([[0],[0],[0],[1]])) 
print(a)
b= get_projection_matrix()@a
b /= b[2,0]
print(b)
