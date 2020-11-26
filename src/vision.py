def rotateX( theta, coords):
    matrix = np.array([[1, 0, 0], [0, np.cos(theta), -1 * np.cos(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(matrix, coords)

def euclideanNorm( vector):
    total = 0
    for val in vector:
      total = total + pow(val, 2)
    return sqrt(total)

def angleBetweenVectors( v1, v2):
    return acos(np.dot(v1,v2) / (self.euclideanNorm(v1) * self.euclideanNorm(v2)))

def projectionOntoPlane( vector, planeNormal):
    return vector - self.projection(vector, planeNormal)

def projection( v1, v2): #Projects v1 onto v2
    return np.multiply((np.dot(v1,v2) / pow(self.euclideanNorm(v2),2)), v2)
