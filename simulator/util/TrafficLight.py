from .Actor import Actor
import cv2
import numpy as np

class TrafficLight(Actor):

    def __init__(self):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        """
        super().__init__()
        self.vertices_L = np.array([[-20, 0, -20, 1], #x, y, z   x increases to right, y up, z forward
                                    [-20, 0,  20, 1],
                                    [20, 0,  20, 1],
                                    [20, 0,  -20, 1]]).T
        self.vertices_W = self.T.dot(self.vertices_L)

