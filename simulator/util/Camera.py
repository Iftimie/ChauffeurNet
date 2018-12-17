from .Actor import Actor
from .transform.util import transformation_matrix
import numpy as np

class Camera(Actor):

    def __init__(self, cam_config):
        """
        :param cam_config: dictionary containing image width, image height, focal length in centimeters, pixel_width in centimeters
        :param x, y, z, roll, yaw, pitch in world coordinates
        T = transformation matrix in world coordinates R * t
        """
        self.K = self.create_K(cam_config)
        self.T = np.eye(4)
        self.C = self.create_cammera_matrix(self.T, self.K)

    def create_cammera_matrix(self, T, K):
        """
        Create camera matrix. it will be a 4x4 matrix
        T defines the camera rotation and translation in world coordinate system.
        we need a matrix that will transform points from world coordinates to camera coordinates in order to project them
        that matrix will do the inverse of translation followed by inverse of rotation followed by camera matrix
        """
        C = K.dot(np.linalg.inv(T)[:3,:])
        return C

    def create_K(self, cam_config):
        img_w = cam_config["img_w"]
        img_h = cam_config["img_h"]
        f_cm  = cam_config["f_cm"]
        pixel_width = cam_config["pixel_width_cm"]

        fx = f_cm / pixel_width
        fy = f_cm / pixel_width
        cx = img_w / 2
        cy = img_h / 2

        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy

        return K


    #@Override
    def set_transform(self, x = 0,y = 0,z = 0,roll = 0, yaw = 0, pitch = 0):
        self.T = transformation_matrix(x, y, z, roll, yaw, pitch)
        self.C = self.create_cammera_matrix(self.T, self.K)

