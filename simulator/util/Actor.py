import numpy as np
from .transform.util import transformation_matrix, params_from_tansformation
import cv2
from math import radians

class Actor:

    # global variable
    delta = 10

    def __init__(self):
        """
        self.c is colour
        self.T is transformation matrix
        self.vertices_L points in local coordinate system
        self.vertices_W points in global coordinate system
        self.delta   when updating position, delta is the distance moved
        """
        self.c = (100,100,100)
        self.T = np.eye(4)
        self.vertices_L = np.array([[0, 0, 0, 1]]).T #vertices defined in local coordinate system
        self.vertices_W = self.T.dot(self.vertices_L)

        self.DRAW_POLYGON = True
        self.render_thickness = 5

        self.is_active = False
        self.move_by_mouse = False

        pass

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
            if self.DRAW_POLYGON:
                image = cv2.fillPoly(image, [pts], color=self.c)
            else:
                image = cv2.polylines(image, [pts], False, color=self.c,thickness= self.render_thickness)
        return image

    def set_transform(self, x=None, y=None, z=None, roll=None, yaw=None, pitch=None):
        """
        values in world coordinates

        X axis positive to the right, negative to the right
        Y axis positive up, negative down
        Z axis positive forward, negative backwards
        :param roll:  angle in radians around Z axis
        :param yaw:   angle in radians around Y axis
        :param pitch: angle in radians around X axis
        """
        current_params = params_from_tansformation(self.T)
        if x is None: x = current_params[0]
        if y is None: y = current_params[1]
        if z is None: z = current_params[2]
        if roll is None: roll = current_params[3]
        if yaw is None: yaw = current_params[4]
        if pitch is None: pitch = current_params[5]

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
        self.is_active = False

    def set_active(self):
        self.c = (10,10,240)
        self.is_active = True

    def update_delta(self, key):
        if key == 122:
            Actor.delta +=10
        if key == 120:
            Actor.delta -=10
            Actor.delta = Actor.delta if Actor.delta>10 else 10

    def simulate(self, pressed_key, mouse):
        self.interpret_key(pressed_key)
        self.interpret_mouse(mouse)

    def interpret_key(self, key):
        if self.is_active:
            if key in [122, 120]:
                self.update_delta(key)
            if key in [119, 100, 115, 97, 43, 45, 113, 101]:
                self.move_actor(key)

    def interpret_mouse(self, mouse):
        """
        :param mouse: tuple of 2 coordinates, x and y on the screen
        :return: nothing
        """
        if self.move_by_mouse==True and len(mouse) == 3:
            self.set_transform(x = mouse[0],y = mouse[1],z = mouse[2])
        pass

    def move_actor(self, key):
        x, y, z, roll, yaw, pitch = self.get_transform()
        if key == 119:
            z +=Actor.delta
        if key == 100:
            x +=Actor.delta
        if key == 115:
            z -=Actor.delta
        if key == 97:
            x -=Actor.delta
        if key == 43:
            y += Actor.delta
        if key == 45:
            y -= Actor.delta
        if key == 113:
            yaw -= radians(Actor.delta / 10)
        if key == 101:
            yaw += radians(Actor.delta / 10)
        self.set_transform(x, y, z, roll, yaw, pitch)

    def to_h5py(self):
        vect_T = np.reshape(self.T, (-1))
        vect_vertices_L = np.reshape(self.vertices_L, (-1))
        return np.hstack((vect_T, vect_vertices_L))

    def from_h5py(self, vect):
        s_T_matrix = 16
        self.T = np.reshape(vect[:s_T_matrix], (4,4))
        s_points = self.vertices_L.shape[1] * 4 #for each point we have 4 coordinates
        self.vertices_L = np.reshape(vect[s_T_matrix:s_T_matrix+s_points], (4, -1))
        self.vertices_W = self.T.dot(self.vertices_L)

    def toggle_move_by_mouse(self):
        if self.move_by_mouse == False:
            self.move_by_mouse = True
        else:
            self.move_by_mouse = False