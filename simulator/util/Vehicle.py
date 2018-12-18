from .Actor import Actor
import numpy as np
from  math import *
import cv2

class Vehicle(Actor):

    def __init__(self, camera = None):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        """
        super().__init__()
        self.vertices_L = np.array([[-30, 0, -60, 1], #x, y, z   x increases to right, y up, z forward
                                    [-30, 0,  60, 1],
                                    [30, 0,  60, 1],
                                    [30, 0,  -60, 1]]).T
        self.next_locations = np.zeros((4,60), np.float32)
        self.next_locations[3,:] = 1
        self.vertices_W = self.T.dot(self.vertices_L)
        self.camera = camera

        #Kinematic model and variables as in:
        #https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
        self.turn_angle = 0        #alpha
        self.wheel_base = 120 #W length of car
        self.speed = 0        #d
        self.delta = 1        # unit of time here (unlike in World editor which is displacement. TODO change this

    #@Override
    def interpret_key(self, key):
        if key == 119:
            self.speed += 1
        if key == 115:
            self.speed -= 1
        if key == 100:
            self.turn_angle += 0.0174533
        if key == 97:
            self.turn_angle -= 0.0174533

        self.turn_angle = max(-0.785398, min(self.turn_angle, 0.785398))

        #TODO check what happens when speed is less than 0

    #@Override
    def render(self, image, C):
        super(Vehicle, self).render(image, C)
        if self.next_locations.shape[1] > 1:
            x, y = C.project(self.next_locations)
            for i in range(0, len(x)):
                image = cv2.circle(image, (x[i], y[i]), 1, (0, 0, 255), -1)
        return image

    # #@Override
    # def set_transform(self, x=0, y=0, z=0, roll=0, yaw=0, pitch=0):
    #     super(Vehicle, self).set_transform(x, y, z, roll, yaw, pitch)
    #     self.vertices_W = self.T.dot(self.vertices_L)
    #     return

    def simulate(self):
        x, y, z, roll, yaw, pitch = self.get_transform()

        distance = self.speed * self.delta

        if abs(self.turn_angle) > 0.0001:
            turn_angle_radians = self.turn_angle
            yaw_radians = radians(yaw)
            tan_steering = tan(turn_angle_radians)
            beta_radians = (distance / self.wheel_base) * tan_steering
            beta_degrees = degrees(beta_radians)
            r = self.wheel_base / tan_steering
            sinh, sinhb = sin(yaw_radians), sin(yaw_radians + beta_radians)
            cosh, coshb = cos(yaw_radians), cos(yaw_radians + beta_radians)

            z += -r * sinh + r* sinhb
            x += r * cosh - r* coshb
            yaw += beta_degrees

            tmp_z = z
            tmp_x = x
            tmp_yaw = yaw

            ##################################################
            # next location prediction
            for i in range (self.next_locations.shape[1]):
                turn_angle_radians = self.turn_angle
                yaw_radians = radians(tmp_yaw)
                tan_steering = tan(turn_angle_radians)
                beta_radians = (distance / self.wheel_base) * tan_steering
                beta_degrees = degrees(beta_radians)
                r = self.wheel_base / tan_steering
                sinh, sinhb = sin(yaw_radians), sin(yaw_radians + beta_radians)
                cosh, coshb = cos(yaw_radians), cos(yaw_radians + beta_radians)
                tmp_z += -r * sinh + r * sinhb
                tmp_x += r * cosh - r * coshb
                tmp_yaw += beta_degrees
                self.next_locations[0, i] = tmp_x
                self.next_locations[2, i] = tmp_z
            #################################################
        else:
            z += distance * cos(radians(yaw))
            x += distance * sin(radians(yaw))
            tmp_z = z
            tmp_x = x
            for i in range(self.next_locations.shape[1]):
                tmp_z += distance * cos(radians(yaw))
                tmp_x += distance * sin(radians(yaw))
                self.next_locations[0, i] = tmp_x
                self.next_locations[2, i] = tmp_z

        self.set_transform(x, y, z, roll, yaw, pitch)

        x_c, y_c, z_c, roll_c, yaw_c, pitch_c = self.camera.get_transform()
        self.camera.set_transform(x, y_c, z, roll_c, yaw, pitch_c)