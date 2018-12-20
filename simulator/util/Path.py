from .Actor import Actor
import numpy as np
from .transform.util import params_from_tansformation

class Path(Actor):

    def __init__(self, all_states):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        all_states should be a list of states received from data generation...this is not too modular...but...TODO refactor this
        """
        super().__init__()
        self.vertices_L = []
        for state in all_states:
            coordinates6DOF = params_from_tansformation(state[0])
            coordinates_translation = np.array([[coordinates6DOF[0],coordinates6DOF[1],coordinates6DOF[2], 1]]).T
            self.vertices_L.append(coordinates_translation) # vehicle transform

        self.vertices_L= np.hstack(tuple(self.vertices_L))
        self.vertices_W = self.T.dot(self.vertices_L)
        self.render_thickness = 40
        self.DRAW_POLYGON = False
        self.c = (255,0,0)


