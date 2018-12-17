from .Actor import Actor
from .transform.util import rot_y
import math
import numpy as np

class CurvedLaneMarking(Actor):

    def __init__(self, arc_degree, radius):
        """
        """
        super().__init__()
        self.vertices_L = np.zeros((4, 20))
        self.vertices_L[3,:] = 1
        self.vertices_L[0,:10] +=radius
        self.vertices_L[0,10:] +=radius - 10

        angles = np.linspace(0, arc_degree, 10)
        for i in range(len(angles)):
            deg = angles[i]
            vert = self.vertices_L[:,[i]]
            R = rot_y(math.radians(deg))
            vert = R.dot(vert)
            self.vertices_L[:,[i]] = vert

        angles = np.linspace(arc_degree, 0, 10)
        for i in range(len(angles)):
            deg = angles[i]
            vert = self.vertices_L[:,[i+10]]
            R = rot_y(math.radians(deg))
            vert = R.dot(vert)
            self.vertices_L[:,[i+10]] = vert

        center = np.mean(self.vertices_L, axis = 1, keepdims=True)
        self.vertices_L[:3,:] -= center[:3,:]
        self.vertices_W = self.T.dot(self.vertices_L)