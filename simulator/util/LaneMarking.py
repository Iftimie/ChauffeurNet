from .Actor import Actor
import numpy as np

class LaneMarking(Actor):

    def __init__(self):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        """
        super().__init__()
        self.vertices_L = np.array([[-5, 0, -100, 1], #x, y, z   x increases to right, y up, z forward
                                    [-5, 0,  100, 1],
                                    [5, 0,  100, 1],
                                    [5, 0,  -100, 1]]).T
        self.vertices_W = self.T.dot(self.vertices_L)


