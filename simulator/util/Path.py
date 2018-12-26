from .Actor import Actor
import numpy as np
from .transform.util import params_from_tansformation
import cv2
import random
from math import *
from config import Config

class Path(Actor):

    def __init__(self, all_states):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        all_states should be a list of states received from recording step
        """
        super().__init__()
        self.vertices_L = []
        for state in all_states:
            coordinates6DOF = params_from_tansformation(state["vehicle"]["T"])
            coordinates_translation = np.array([[coordinates6DOF[0],coordinates6DOF[1],coordinates6DOF[2], 1]]).T
            self.vertices_L.append(coordinates_translation) # vehicle transform

        self.vertices_L= np.hstack(tuple(self.vertices_L))
        self.vertices_W = self.T.dot(self.vertices_L)
        self.render_thickness = 40
        self.DRAW_POLYGON = False
        self.c = (180,180,180)

        #dropout part
        self.dropout_idx = None
        self.dropout_cached_vertices = None
        self.sign = 1

    def render(self, image, C, path_idx):
        """
        :param image: image on which this actor will be renderd on
        :param C:     camera matrix
        :param path_idx: since the path contains all indices, we don't want to render them all cause it will cause a mess
        we want to render only the next positions
        :return:      image with this object renderd
        """
        if self.vertices_W.shape[1] > 1:
            selected_for_projection = self.vertices_W[:,path_idx:path_idx+ Config.path_future_len]
            x, y = C.project(selected_for_projection)
            pts = np.array([x, y]).T
            pts = pts.reshape((-1, 1, 2))
            if self.DRAW_POLYGON:
                image = cv2.fillPoly(image, [pts], color=self.c)
            else:
                thick = int(ceil(self.render_thickness / Config.r_ratio))
                image = cv2.polylines(image, [pts], False, color=self.c,thickness= thick)

        if self.dropout_cached_vertices  is not None:
            x, y = C.project(self.dropout_cached_vertices)
            for i in range(0, len(x)):
                image = cv2.circle(image, (x[i], y[i]), 0, (0, 0, 255),0 )
        return image

    def project_future_poses(self, C, path_idx):

        if self.vertices_W.shape[1] > 1:
            selected_for_projection = self.vertices_W[:, [path_idx]]
            x, y = C.project(selected_for_projection)
            if len(x) !=1:
                raise ValueError("too many points. there is a problem")
            return [x,y]

    def apply_dropout(self, path_idx):

        num_frames = 70
        total = num_frames * 2 + 1
        if path_idx % 300 ==0:
            self.sign *=-1


        if not( path_idx > num_frames and path_idx < self.vertices_W.shape[1] - num_frames):return

        #cache modifications
        self.dropout_cached_vertices = self.vertices_W[:,path_idx-num_frames:path_idx+num_frames +1].copy()
        print (self.dropout_cached_vertices.shape[1])
        self.dropout_idx = path_idx

        #apply modification

        heading = self.dropout_cached_vertices[:,-1] - self.dropout_cached_vertices[:,0]
        length = np.linalg.norm(heading)
        half_len = length / 2
        # https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector
        perpendicular = np.array([[heading[2], heading[1], -heading[0]]]).T
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        perpendicular *= self.sign
        tiled_perpendicular = np.tile(perpendicular, (1,total))

        sigma = (1/3 * half_len)
        amplitude = random.uniform(0.5/10, 2/10) * length
        weights = np.linspace(-half_len,half_len,total)

        gauss = lambda x, mu, sig : np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        weights = gauss(weights,0,sigma) * amplitude

        tiled_perpendicular = tiled_perpendicular * weights
        self.dropout_cached_vertices[:3,:] +=tiled_perpendicular[:3,:]

        return

