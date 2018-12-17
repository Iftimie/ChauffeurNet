from .Actor import Actor
from .transform.util import transformation_matrix
import numpy as np
import cv2

class LaneMarking(Actor):

    def __init__(self):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        """
        super().__init__()
        self.transform = np.eye(4)
        self.vertices_L = np.array([[-5, 0, -100, 1], #x, y, z   x increases to right, y up, z forward
                                    [-5, 0,  100, 1],
                                    [5, 0,  100, 1],
                                    [5, 0,  -100, 1]]).T
        self.vertices_W = self.transform.dot(self.vertices_L)

    #@Override
    def set_transform(self, x = 0,y = 0,z = 0,roll = 0, yaw = 0, pitch = 0):
        self.transform = transformation_matrix(x, y, z, roll, yaw, pitch)
        self.vertices_W = self.transform.dot(self.vertices_L)
        return


    #@Override
    def render(self,image, C):


        x, y = C.project(self.vertices_W)
        image = cv2.line(image, pt1= (x[0],y[0]), pt2=(x[1],y[1]), color = self.c,thickness=1)
        image = cv2.line(image, pt1=(x[1],y[1]), pt2=(x[2],y[2]), color = self.c,thickness=1)
        image = cv2.line(image, pt1=(x[2],y[2]), pt2=(x[3],y[3]), color = self.c,thickness=1)
        image = cv2.line(image, pt1=(x[3],y[3]), pt2=(x[0],y[0]), color = self.c,thickness=1)
        return image