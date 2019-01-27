from simulator.util.Actor import Actor
import numpy as np
import cv2
from math import ceil
from config import Config

class LaneMarking(Actor):

    def __init__(self):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        """
        super().__init__()
        self.vertices_W = None
        self.c = (150,150,150)

    def render(self, image, C):
        """
        :param image: image on which this actor will be renderd on
        :param C:     camera matrix
        :return:      image with this object renderd
        """
        if self.vertices_W.shape[1] > 1:
            x, y = C.project(self.vertices_W)
            pts = np.array([x, y]).T
            pts = pts.reshape((-1, 1, 2))
            thick = int(ceil(self.render_thickness / Config.r_ratio))
            image = cv2.polylines(image, [pts], False, color=self.c,thickness= thick)
        return image

