from .Actor import Actor
import numpy as np
from  math import *
import cv2
from scipy.interpolate import interp1d
from .transform.util import rot_y

class Vehicle(Actor):

    def __init__(self, camera = None, play = True):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        play: if playing, the colours and the shape of the future positions will be different
        """
        super().__init__()
        self.vertices_L = np.array([[-30, 0, -60, 1], #x, y, z   x increases to right, y up, z forward
                                    [-30, 0,  60, 1],
                                    [30, 0,  60, 1],
                                    [30, 0,  -60, 1]]).T
        self.next_locations = np.zeros((4,15), np.float32)
        self.next_locations[3,:] = 1
        self.vertices_W = self.T.dot(self.vertices_L)
        self.camera = camera

        #Kinematic network and variables as in:
        #https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
        self.turn_angle = 0        #alpha in radians
        self.wheel_base = 120 #W length of car
        self.speed = 0        #d
        self.delta = 1        # unit of time here (unlike in World editor which is displacement. TODO change this

        self.range_angle = (-0.785398, 0.785398)

        if play:
            self.render_radius = 2
            self.render_thickness = -1
        else:
            self.render_radius = 15
            self.render_thickness = 5

        self.is_active = True



    #@Override
    def render(self, image, C):
        super(Vehicle, self).render(image, C)
        if self.next_locations.shape[1] > 1:
            x, y = C.project(self.next_locations)
            for i in range(0, len(x)):
                image = cv2.circle(image, (x[i], y[i]), self.render_radius, (0, 0, 255),self.render_thickness)
        return image

    # @Override
    def from_h5py(self, vect):
        super(Vehicle, self).from_h5py(vect)
        s_T_matrix = 16
        s_points = self.vertices_L.shape[1] * 4
        offset = s_T_matrix +s_points
        self.turn_angle = vect[offset]  # alpha in radians
        self.speed = vect[offset + 1]  # d

    #Override
    def to_h5py(self):
        vect = super(Vehicle, self).to_h5py()
        vect = np.hstack((vect, np.array([self.turn_angle, self.speed])))
        return vect

    def kinematic_model(self, z, x, yaw, delta):
        distance = self.speed * delta
        tan_steering = tan(self.turn_angle)
        beta_radians = (distance / self.wheel_base) * tan_steering
        r = self.wheel_base / tan_steering
        sinh, sinhb = sin(yaw), sin(yaw + beta_radians)
        cosh, coshb = cos(yaw), cos(yaw + beta_radians)
        z += -r * sinh + r * sinhb
        x += r * cosh - r * coshb
        yaw += beta_radians
        return z, x, yaw

    def linear_model(self, z, x, yaw, delta):
        distance = self.speed * delta
        z += distance * cos(yaw)
        x += distance * sin(yaw)
        return z, x

    # @Override
    def interpret_key(self, key):
        if self.is_active:
            if key == 119:
                self.speed += 1
            if key == 115:
                self.speed -= 1
            if key == 100:
                self.turn_angle += 0.0174533  # 1 degrees
            if key == 97:
                self.turn_angle -= 0.0174533
            self.turn_angle = max(self.range_angle[0], min(self.turn_angle, self.range_angle[1]))  # 45 degrees

    # @Override
    def interpret_mouse(self, mouse):
        if self.is_active and mouse is not None:
            min_x = 150
            max_x = 640 - 150
            x_pos = max(min_x, min(mouse[0], max_x))  # 45 degrees
            m_func = interp1d([min_x, max_x], [self.range_angle[0], self.range_angle[1]])
            self.turn_angle = m_func(x_pos)

    def simulate(self, key_pressed, mouse):
        self.interpret_key(key_pressed)
        self.interpret_mouse(mouse)

        x, y, z, roll, yaw, pitch = self.get_transform()

        if abs(self.turn_angle) > 0.0001: # is the car turning?

            z, x, yaw = self.kinematic_model(z,x, yaw, self.delta)

            tmp_z,tmp_x, tmp_yaw = z, x, yaw
            #next location prediction
            for i in range (self.next_locations.shape[1]):
                tmp_z, tmp_x, tmp_yaw = self.kinematic_model(tmp_z, tmp_x, tmp_yaw, self.delta * 4)
                self.next_locations[0, i] = tmp_x
                self.next_locations[2, i] = tmp_z
        else:
            z, x = self.linear_model(z, x, yaw, self.delta)
            tmp_z, tmp_x = z, x
            #next location prediction
            for i in range(self.next_locations.shape[1]):
                tmp_z, tmp_x = self.linear_model(tmp_z, tmp_x, yaw , self.delta * 4)
                self.next_locations[0, i] = tmp_x
                self.next_locations[2, i] = tmp_z

        self.set_transform(x, y, z, roll, yaw, pitch)

        x_c, y_c, z_c, roll_c, yaw_c, pitch_c = self.camera.get_transform()
        displacement_vector = np.array([[0,0,-200,1]]).T
        displacement_vector = rot_y(yaw - pi).dot(displacement_vector)
        x +=displacement_vector[0]
        z +=displacement_vector[2]
        self.camera.set_transform(x, y_c, z, roll_c, yaw, pitch_c)