import numpy as np
from .transform.util import transformation_matrix, params_from_tansformation
import cv2

class Actor:

    def __init__(self):
        """
        self.c is colour
        """
        self.c = (100,100,100)
        self.T = np.eye(4)
        self.vertices_L = np.array([[0, 0, 0, 1]]).T #vertices defined in local coordinate system
        self.vertices_W = self.T.dot(self.vertices_L)
        pass

    def render(self, image, C):
        """
        :param image: image on which this actor will be renderd on
        :param C:     camera matrix
        :return:      image with this object renderd
        """
        if self.vertices_W.shape[1] > 1:
            x, y = C.project(self.vertices_W)
            for i in range(1, len(x)):
                prev = i-1
                curr = i
                image = cv2.line(image, pt1=(x[prev], y[prev]), pt2=(x[curr], y[curr]), color=self.c, thickness=1)
            image = cv2.line(image, pt1=(x[curr], y[curr]), pt2=(x[0], y[0]), color=self.c, thickness=1)
        return image

    def set_transform(self, x=0, y=0, z=0, roll=0, yaw=0, pitch=0):
        """
        values in world coordinates

        X axis positive to the right, negative to the right
        Y axis positive up, negative down
        Z axis positive forward, negative backwards
        :param roll:  angle in degrees around Z axis
        :param yaw:   angle in degrees around Y axis
        :param pitch: angle in degrees around X axis
        """
        self.T = transformation_matrix(x, y, z, roll, yaw, pitch)
        self.vertices_W = self.T.dot(self.vertices_L)
        return


    def get_transform(self):
        """
        return the transformation parameters as x, y, z, roll, yaw, pitch
        :return:
        """
        return params_from_tansformation(self.T)

    def set_color(self, c):
        self.c = c

    def set_inactive(self):
        self.c = (100,100,100)

    def set_active(self):
        self.c = (10,10,240)