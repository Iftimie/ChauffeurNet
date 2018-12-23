from .Actor import Actor
import numpy as np
from  math import *
import cv2
from scipy.interpolate import interp1d
from .transform.util import rot_y
import math
from simulator.util.transform.util import params_from_tansformation

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
        self.c = (200,200,200)

        self.vertices_L = np.array([[-30, 0, -60, 1], #x, y, z   x increases to right, y up, z forward
                                    [-30, 0,  60, 1],
                                    [30, 0,  60, 1],
                                    [30, 0,  -60, 1]]).T
        self.next_locations_by_steering = np.zeros((4,15), np.float32)
        self.next_locations_by_steering[3,:] = 1
        self.past_locations = []
        self.vertices_W = self.T.dot(self.vertices_L)
        self.camera = camera
        self.displacement_vector = np.array([[0, 0, -200, 1]]).T

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
        self.render_next_locations_by_steering = False

        #Options about the past
        self.render_past_locations = False
        self.render_past_locations_thickness = 2
        self.render_past_locations_radius = 2
        self.num_past_poses = 40
        self.num_skip_poses = 5
        #TODO do not forget that the delta time must be the same in all settings

    def render_next_locations_by_steering_func(self, image, C):
        if self.next_locations_by_steering.shape[1] > 1:
            x, y = C.project(self.next_locations_by_steering)
            for i in range(0, len(x)):
                thick = int(ceil(self.render_thickness / self.ratio))
                radius = int(ceil(self.render_radius / self.ratio))
                image = cv2.circle(image, (x[i], y[i]), radius , (0, 0, 255),thick)
        return image

    def render_past_locations_func(self, image, C):
        if len(self.past_locations) > 0:
            array_past_locations = np.array(self.past_locations[::self.num_skip_poses]).T
            x, y = C.project(array_past_locations)
            for i in range(0, len(x)):
                thick = int(ceil(self.render_past_locations_thickness / self.ratio))
                radius = int(self.render_past_locations_radius / self.ratio)
                image = cv2.circle(image, (x[i], y[i]), radius, (0, 0, 255),thick )
        return image

    #@Override
    def render(self, image, C):
        image = super(Vehicle, self).render(image, C)
        if self.render_next_locations_by_steering == True:
            image = self.render_next_locations_by_steering_func(image,C)
        if self.render_past_locations == True:
            image = self.render_past_locations_func(image,C)

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

    def append_past_location(self, past_location):
        if len(self.past_locations) > self.num_past_poses:
            self.past_locations.pop(0)
        if len(past_location) == 3: # if it is received as a tuple
            self.past_locations.append([past_location[0],past_location[1],past_location[2],1])
        elif len(past_location) == 4: # if it is received as a transformation matrix
            x, y, z, roll, yaw, pitch = params_from_tansformation(past_location) #TODO maybe refactor this to be not depend directly on the outer package
            self.past_locations.append([x, y, z,1])

    def update_parameters(self):
        x, y, z, roll, yaw, pitch = self.get_transform()

        self.append_past_location((x,y,z))

        if abs(self.turn_angle) > 0.0001:  # is the car turning?

            z, x, yaw = self.kinematic_model(z, x, yaw, self.delta)

            tmp_z, tmp_x, tmp_yaw = z, x, yaw
            # next location prediction
            for i in range(self.next_locations_by_steering.shape[1]):
                tmp_z, tmp_x, tmp_yaw = self.kinematic_model(tmp_z, tmp_x, tmp_yaw, self.delta * 4)
                self.next_locations_by_steering[0, i] = tmp_x
                self.next_locations_by_steering[2, i] = tmp_z
        else:
            z, x = self.linear_model(z, x, yaw, self.delta)
            tmp_z, tmp_x = z, x
            # next location prediction
            for i in range(self.next_locations_by_steering.shape[1]):
                tmp_z, tmp_x = self.linear_model(tmp_z, tmp_x, yaw, self.delta * 4)
                self.next_locations_by_steering[0, i] = tmp_x
                self.next_locations_by_steering[2, i] = tmp_z

        self.set_transform(x, y, z, roll, yaw, pitch)

        x_c, y_c, z_c, roll_c, yaw_c, pitch_c = self.camera.get_transform()
        rotated_displacement_vector = rot_y(yaw - pi).dot(self.displacement_vector)
        x += rotated_displacement_vector[0]
        z += rotated_displacement_vector[2]
        self.camera.set_transform(x, y_c, z, roll_c, yaw, pitch_c)

    def simulate(self, key_pressed, mouse):
        self.interpret_key(key_pressed)
        self.interpret_mouse(mouse)
        self.update_parameters()


    def simulate_given_waypoint(self, x,z,yaw, mouse):
        """
        This method should not modify the speed, it will update the position and the orientation, given the desired location and orientation
        """
        if self.speed == 0:
            return
        x_c, y_c, z_c, roll_c, yaw_c, pitch_c = self.get_transform()

        #future position if steering would be straight
        z_f, x_f = self.linear_model(z_c, x_c, yaw_c, self.delta)
        z_h, x_h = z_f - z_c, x_f - x_c             #the heading vector
        z_d, x_d = z   - z_c, x   - x_c                 #the desired heading vector


        heading = np.array([x_h, 0, z_h])
        heading = heading / np.linalg.norm(heading)
        desired = np.array([x_d, 0, z_d])
        desired = desired / np.linalg.norm(desired)
        delta_angle = np.arccos(heading.dot(desired))
        cross = np.cross(heading, desired)

        plane_normal = np.array([0,-1,0])

        if plane_normal.dot(cross) > 0:
            delta_angle = -delta_angle

        next_turn_angle = math.atan(delta_angle * self.wheel_base/self.speed) #the desired turn angle that would take us from current yaw to desired yaw in an instant
        next_turn_angle = max(self.range_angle[0], min(next_turn_angle* 2, self.range_angle[1]))  # clamp the desired turn angle since that is not possible
        #I don't fucking know why it has to be multiplied by 2 but it works
        self.turn_angle = next_turn_angle

        self.update_parameters()
        # self.interpret_mouse(mouse)

        # # Don't update the position.
        # dx,dz = x-x_c, z-z_c
        # mag = sqrt(dx**2 + dz**2)
        # if self.speed == 0:return
        # step = mag / self.speed
        # x_n = x_c + dx/step
        # z_n = z_c + dz/step
        # dyaw = yaw - yaw_c
        # self.turn_angle = math.atan(dyaw * self.wheel_base/self.speed) #the desired turn angle that would take us from current yaw to desired yaw in an instant
        # self.turn_angle = max(self.range_angle[0], min(self.turn_angle, self.range_angle[1]))  # clamp the desired turn angle since that is not possible
        # yaw_n = yaw_c + dyaw
        # # self.set_transform(x=x_n,z=z_n,yaw=yaw_n)
        pass
