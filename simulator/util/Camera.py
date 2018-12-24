from .Actor import Actor
import numpy as np

class Camera(Actor):
    def __init__(self, cam_config = None, in_res= (144,192)):
        """
        :param cam_config: dictionary containing image width, image height, focal length in centimeters, pixel_width in centimeters
        :param x, y, z, roll, yaw, pitch in world coordinates
        T = transformation matrix in world coordinates R * t
        """
        super().__init__()
        if in_res == None:
            raise ValueError("in_res must not be None")
        if cam_config == None:
            #TODO pixel width must also be resized I think
            #TODO ratio currently is 6.666666, any drawing with opencv should also be done in this way
            self.cam_config = {"img_w": 192, "img_h": 144, "f_cm": 0.238 /self.ratio, "pixel_width_cm": 0.0003 }
        else:
            self.cam_config = cam_config
        self.K = self.create_K_4x4(self.cam_config)
        self.set_transform(x=0, y=-1500, z=0, roll=0, yaw=0, pitch=-1.5708)

        self.project = self.project_perspective

    def create_internal_cam_matrix(self, in_res = None):
        if in_res == None:
            raise ValueError("in_res must not be None")
        self.cam_config["img_h"] = in_res[0]
        self.cam_config["img_w"] = in_res[1]
        self.K = self.create_K_4x4(self.cam_config)
        x,y,z,roll,yaw,pitch = self.get_transform()
        self.set_transform(x=x, y=y, z=z, roll=roll, yaw=yaw, pitch=pitch)

    def create_cammera_matrix(self, T, K):
        """
        Create camera matrix. it will be a 4x4 matrix
        T defines the camera rotation and translation in world coordinate system.
        we need a matrix that will transform points from world coordinates to camera coordinates in order to project them
        that matrix will do the inverse of translation followed by inverse of rotation followed by camera matrix
        """
        C = K.dot(np.linalg.inv(T)[:3,:])
        return C

    def create_cammera_matrix4x4(self, T, K):
        """
        Create camera matrix. it will be a 4x4 matrix
        T defines the camera rotation and translation in world coordinate system.
        we need a matrix that will transform points from world coordinates to camera coordinates in order to project them
        that matrix will do the inverse of translation followed by inverse of rotation followed by camera matrix

        Maybe using a matrix with smaller values, it will reduce some noise in projection???

        Found some hints at the following links. It is a great course, and I addapted for my usage. The camera projection matrix there is not the same as in here
        http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
        https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
        http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/virtual-pinhole-camera-model
        """
        C = K.dot(np.linalg.inv(T))
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

    def create_K_4x4(self,cam_config):
        """

        Maybe using a matrix with smaller values, it will reduce some noise in projection???

        Found some hints at the following links. It is a great course, and I addapted for my usage. The camera projection matrix there is not the same as in here
        http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
        https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
        http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/virtual-pinhole-camera-model
        :param cam_config:
        :return:
        """
        img_w = cam_config["img_w"]
        img_h = cam_config["img_h"]
        f_cm = cam_config["f_cm"]
        pixel_width = cam_config["pixel_width_cm"]

        fx = f_cm / pixel_width
        fy = f_cm / pixel_width
        cx = img_w / 2
        cy = img_h / 2

        K = np.eye(4)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 3] = 0
        K[3, 1] = 1

        return K

    #@Override
    def set_transform(self, x=None, y=None, z=None, roll=None, yaw=None, pitch=None):
        super(Camera, self).set_transform(x ,y ,z ,roll , yaw , pitch )
        self.C = self.create_cammera_matrix4x4(self.T, self.K)

    def project_ortographic(self, vertices):
        homogeneous_vertices = self.C.dot(vertices)
        homogeneous_vertices /= homogeneous_vertices[2, :]
        homogeneous_vertices += self.center
        v = homogeneous_vertices.astype(np.int32)
        x = v[0, :]
        y = v[1, :]
        return x, y

    def project_perspective(self, vertices):
        #vertices have shape 4xN.
        #C is 3x4 matrix
        homogeneous_vertices = self.C.dot(vertices)
        # divide u, v by z
        projected_vertices = np.round(homogeneous_vertices / homogeneous_vertices[2, :])
        v = projected_vertices.astype(np.int32)
        x = v[0, :]
        y = v[1, :]
        return x, y

    def toggle_projection(self):
        if self.project == self.project_perspective:
            self.K = np.eye(3)
            self.center = np.array([[self.cam_config["img_w"] / 2],
                                    [self.cam_config["img_h"] / 2],
                                    [0.0]])
            self.project = self.project_ortographic
        else:
            self.K = self.create_K(self.cam_config)
            self.project = self.project_perspective
        self.C = self.create_cammera_matrix4x4(self.T, self.K)

    #@Override
    def set_active(self):
        self.is_active = True
        print ("Reached camera. No action for color")

    #@Override
    def interpret_key(self, key):
        super(Camera, self).interpret_key(key)
        if key in [122, 120]:
            self.update_delta(key)
        if key in [43, 45]:
            self.move_actor(key)
