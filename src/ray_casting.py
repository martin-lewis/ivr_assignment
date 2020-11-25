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

    a,b,c = 0,0,0
    tx,ty,tz = 0,0,0

    if camera_axis == "x":
        a,b,c = -math.pi/2, math.pi/2, 0
        tx,ty,tz = 0,6-1.25,18
    else:
        a,b,c = -math.pi/2,math.pi/2,math.pi/2
        tx,ty,tz = 0,6-1.25,18

    return np.array([[cos(a)*cos(b)*cos(c) - sin(a)*sin(c),    -cos(a)*cos(b)*cos(c) - sin(a)*cos(c), cos(a)*sin(b) ,   tx],
                    [sin(a)*cos(b)*cos(c) + cos(a)*sin(c), -sin(a)*cos(b)*cos(c) + cos(a)*cos(c),   sin(a)*sin(b),   ty],
                    [-sin(b)*cos(c), sin(b)*sin(c),  cos(b),                          tz]
                    ])

def get_camera_to_world_origin_matrix(camera_axis="x"):
    
    matrix = get_world_to_camera_origin_matrix(camera_axis)

    rot = np.linalg.inv(matrix[:,0:-1])
    trans = rot@matrix[:,-1]
    
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

    start_p_world = (inverse_matrix_transform @ np.array([[0],[0],[0],[1]]))
    end_p_world = inverse_matrix_transform @ direction_h
    return np.array([start_p_world,end_p_world])


def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
                
                
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    
    return pA,pB,np.linalg.norm(pA-pB)

def ray_intersect(ray1,ray2):

    a0 = ray1[0].flatten()
    a1 = ray1[1].flatten()
    b0 = ray2[0].flatten()
    b1 = ray2[1].flatten()
    pA,pB, dist = closestDistanceBetweenLines(a0,a1,b0,b1)
    #TODO: for some reason the above method, flips all components *shrug*, and y with x 
    f = -((pB - pA)/2 + pA)
    x = f[0]
    f[0] = f[1]
    f[1] = x
    return f
    

# # the height of the maximum rectangle seen by camera at distance of 1m
# fh = 2.0 * math.tan(hfov * 0.5)
# print(fh/2)

# middle_ray_x = screen_to_world_ray(
#     np.linalg.inv(get_projection_matrix()),
#     get_camera_to_world_origin_matrix("x"),
#     np.array([[400],[400.76303918]]))
# middle_ray_y = screen_to_world_ray(
#     np.linalg.inv(get_projection_matrix()),
#     get_camera_to_world_origin_matrix("y"),
#     np.array([[400],[400.76303918]]))

# print(middle_ray_x[1] - middle_ray_x[0])
# print(middle_ray_y[1] - middle_ray_y[0])

# print(ray_intersect(middle_ray_y,middle_ray_x))

# print((world_to_screen(get_camera_matrix(camera_axis="x"),np.array([[0],[0],[0],[1]]))))

middle_ray_y = screen_to_world_ray(
    np.linalg.inv(get_projection_matrix()),
    get_camera_to_world_origin_matrix("x"),
    np.array([[600],[400]]))

print(middle_ray_y[1]-middle_ray_y[0])