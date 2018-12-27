from .Actor import Actor
import numpy as np
from .transform.util import params_from_tansformation
import cv2
import random
from math import *
from config import Config

class Path(Actor):

    def __init__(self, all_states, debug ):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        all_states should be a list of states received from recording step
        """
        super().__init__()
        self.vertices_L = []
        self.debug = debug
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

        # Options about the past
        self.render_past_locations = False
        self.render_past_locations_thickness = 8
        self.render_past_locations_radius = 2

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

        if self.dropout_cached_vertices  is not None and self.debug == True:
            selected_for_projection = self.dropout_cached_vertices[:, path_idx:path_idx + Config.num_frames]
            x, y = C.project(selected_for_projection)
            for i in range(0, len(x)):
                image = cv2.circle(image, (x[i], y[i]), 0, (0, 0, 255),0 )
        return image

    def project_future_poses(self, C, current_path_idx, future_waypoint):

        if self.dropout_cached_vertices is None:
            if self.vertices_W.shape[1] > 1:
                selected_for_projection = self.vertices_W[:, [current_path_idx+future_waypoint]]
                x, y = C.project(selected_for_projection)
                if len(x) !=1:
                    raise ValueError("too many points. there is a problem")
                return [x,y]
        else:
            selected_for_projection = self.dropout_cached_vertices[:, [current_path_idx+future_waypoint]]
            x, y = C.project(selected_for_projection)
            if len(x) != 1:
                raise ValueError("too many points. there is a problem")
            return [x, y]

    def apply_dropout(self, path_idx, vehicle):
        if not( path_idx > Config.num_frames and path_idx < self.vertices_W.shape[1] - Config.num_frames):return

        if random.uniform(0.0,1.0) > Config.dropout_prob:
            self.dropout_cached_vertices = None
            return

        total = Config.num_frames * 2 + 1

        #TODO check what to do with the sign...how many times should we alternate?
        # if path_idx % 300 ==0:
        self.sign *=-1

        #cache modifications
        cached_vertices = self.vertices_W[:,path_idx-Config.num_frames:path_idx+Config.num_frames +1].copy()

        #apply modification
        heading = cached_vertices[:,-1] - cached_vertices[:,0]


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
        cached_vertices[:3,:] +=tiled_perpendicular[:3,:]

        yaw = atan2(heading[0], heading[2])
        x,y,z = cached_vertices[0,Config.num_frames],cached_vertices[1,Config.num_frames],cached_vertices[2,Config.num_frames]
        vehicle.set_transform(x=x,y=y,z=z,yaw=yaw)

        self.dropout_cached_vertices = self.vertices_W.copy()
        self.dropout_cached_vertices[:,path_idx-Config.num_frames:path_idx+Config.num_frames +1] = cached_vertices

        return

    def render_past_locations_func(self, image, C, path_idx):

        if self.dropout_cached_vertices is None:
            if self.vertices_W.shape[1] > 1:
                array_past_locations = self.vertices_W[:, path_idx-Config.num_frames:path_idx:Config.num_skip_poses]
                x, y = C.project(array_past_locations)
                for i in range(0, len(x)):
                    thick = int(ceil(self.render_past_locations_thickness / Config.r_ratio))
                    radius = int(self.render_past_locations_radius / Config.r_ratio)
                    image = cv2.circle(image, (x[i], y[i]), radius, (0, 0, 255), thick)
        else:
            array_past_locations = self.dropout_cached_vertices[:, path_idx-Config.num_frames:path_idx:Config.num_skip_poses]
            x, y = C.project(array_past_locations)
            for i in range(0, len(x)):
                thick = int(ceil(self.render_past_locations_thickness / Config.r_ratio))
                radius = int(self.render_past_locations_radius / Config.r_ratio)
                image = cv2.circle(image, (x[i], y[i]), radius, (0, 0, 255),thick )

        return image