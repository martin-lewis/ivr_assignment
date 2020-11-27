from math import pi
import numpy as np

def damped_pseudo_inverse(J,k):
    if(k == 0):
        return np.linalg.pinv(J)
    else:  
        return J.T @ np.linalg.inv(J@J.T + (k**2 * np.eye(J.shape[0])))

def limit_q(q):
    return np.array([x for x in q])
  
# wrap around 
def limit_angle(a):
    if a > pi/2 :
        return -pi/2
    if a < -pi/2:
        return pi/2
