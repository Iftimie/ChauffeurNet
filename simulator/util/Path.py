from .Actor import Actor
import numpy as np
from .transform.util import params_from_tansformation
import cv2

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
        self.c = (180,180,180)
        self.future_positions = 250 # we only want to render the next 200 positions from the current index

        self.num_future_poses = 40
        self.num_skip_poses = 5
        # TODO do not forget that the delta time must be the same in all settings

    def render(self, image, C, path_idx):
        """
        :param image: image on which this actor will be renderd on
        :param C:     camera matrix
        :param path_idx: since the path contains all indices, we don't want to render them all cause it will cause a mess
        we want to render only the next positions
        :return:      image with this object renderd
        """
        if self.vertices_W.shape[1] > 1:
            selected_for_projection = self.vertices_W[:,path_idx:path_idx+self.future_positions]
            x, y = C.project(selected_for_projection)
            pts = np.array([x, y]).T
            pts = pts.reshape((-1, 1, 2))
            if self.DRAW_POLYGON:
                image = cv2.fillPoly(image, [pts], color=self.c)
            else:
                image = cv2.polylines(image, [pts], False, color=self.c,thickness= self.render_thickness)
        return image

    def render_future_poses(self, image, C, path_idx):
        """
        receives a single channel image and put a simple pixel over the location
        """

        if self.vertices_W.shape[1] > 1:
            selected_for_projection = self.vertices_W[:,[path_idx]]
            x, y = C.project(selected_for_projection)
            for i in range(len(x)):
                x_i = x[i]
                y_i = y[i]
                image[x_i,y_i] = (255)
        return image

