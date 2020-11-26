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

    ch = 6-1.25 # height of cameras (above yellow)
    mat = None
    if camera_axis == "x":
        mat = np.array([[0,1,0,0],
                        [0,0,-1,ch],
                        [-1,0,0,18]])
    else:
        mat = np.array([[1,0,0,0],
                        [0,0,-1,ch],
                        [0,1,0,18]])
    return mat

def get_camera_to_world_origin_matrix(camera_axis="x"):
    
    matrix = get_world_to_camera_origin_matrix(camera_axis)

    rot = matrix[:,0:-1].T
    trans = (rot@matrix[:,-1]) * -1

    new_mat = np.zeros((3,4))
    new_mat[:,0:3] = rot
    new_mat[:,-1] = trans

    return new_mat

def get_camera_matrix(camera_axis="x"):
    return get_projection_matrix()@get_world_to_camera_origin_matrix(camera_axis)


def world_to_screen(matrix,point):
    p = matrix @ point
    # projective division
    return p / p[2,0]

def screen_to_world_ray(inverse_projection_matrix,inverse_matrix_transform,point):
    
    if point.shape == (2,):
        point = np.reshape(point,(2,1))
    # append homogenous coordinate to screen point
    h_point = np.append(point,np.array([[1]]),axis=0)
    # find direction of ray in world space
    direction = inverse_projection_matrix@h_point
    direction /= np.linalg.norm(direction)

    # homogenize it
    direction_h = np.append(direction,np.array([[1]]),axis=0)
    # get point in world frame

    camera_origin_world = (inverse_matrix_transform @ np.array([[0],[0],[0],[1]]))
    ray_direction_world = inverse_matrix_transform[:,:-1] @ direction_h[:-1,:]
    return np.array([camera_origin_world,camera_origin_world+ray_direction_world])

def ray_intersect(ray1,ray2):

    a0 = ray1[0].flatten()
    a1 = ray1[1].flatten()
    b0 = ray2[0].flatten()
    b1 = ray2[1].flatten()
    return find_closest_point(a0,a1,b0,b1)
    
def find_closest_point(a0,a1,b0,b1):

    point1 = a0
    point2 = b0 
    dir1 = (a1 - a0)
    dir1 /= np.linalg.norm(dir1)
    dir2 = (b1 - b0)
    dir2 /= np.linalg.norm(dir2)

    slopes_perp = np.cross(dir1, dir2)
    slopes_perp /= np.linalg.norm(slopes_perp)

    RHS = point2 - point1
    LHS = np.array([dir1, -dir2, slopes_perp]).T

    # get t1, t2, t3 (solutions to the system of linear eqs)
    solution = np.linalg.solve(LHS, RHS)

    # get some uncertainty on this measurement
    # e.g. how close together are the two points

    point4 = point2 + (solution[1]*dir2)
    point3 = point1 + (solution[0]*dir1)

    return  point4 + (point3 - point4)/2


# # the height of the maximum rectangle seen by camera at distance of 1m
# fh = 2.0 * math.tan(hfov * 0.5)
# print(fh/2)

# middle_ray_x = screen_to_world_ray(
#     np.linalg.inv(get_projection_matrix()),
#     get_camera_to_world_origin_matrix("x"),
#     np.array([[399],[300]]))
# middle_ray_y = screen_to_world_ray(
#     np.linalg.inv(get_projection_matrix()),
#     get_camera_to_world_origin_matrix("y"),
#     np.array([[399],[300]]))

# print("xray")
# print(middle_ray_x[1] - middle_ray_x[0])
# print("yray")
# print(middle_ray_y[1] - middle_ray_y[0])

# print(ray_intersect(middle_ray_x,middle_ray_y))

# print(get_camera_to_world_origin_matrix(camera_axis="x") @ np.array([[0],[0],[1],[1]]))
# print(get_camera_to_world_origin_matrix(camera_axis="y") @ np.array([[0],[0],[1],[1]]))

# print(get_world_to_camera_origin_matrix(camera_axis="x") @ np.array([[0],[0],[1],[1]]))
# print(get_world_to_camera_origin_matrix(camera_axis="y") @ np.array([[0],[0],[1],[1]]))