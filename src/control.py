import numpy as np

def damped_pseudo_inverse(J,k):
      return J.T @ np.linalg.inv(J@J.T + (k**2 * np.eye(J.shape[0])))

def limit_q(q):
    return np.array([x for x in q])
  
def limit_angle(a):
    return min(max(a,pi/2),-pi/2) 
