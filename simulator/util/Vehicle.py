from simulator.util.Actor import Actor
import numpy as np
from  math import *
import cv2
from scipy.interpolate import interp1d
from simulator.util.transform.util import rot_y
import math
from simulator.util.transform.util import params_from_tansformation
import copy
from config import Config

class Vehicle(Actor):

    def __init__(self, camera = None, play = True, traffic_lights=[]):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        play: if playing, the colours and the shape of the future positions will be different

        traffic_lights can be empty list
        """
        super().__init__()
        self.c = (200,200,200)

        self.vertices_L = np.array([[-30, 0, -60, 1], #x, y, z   x increases to right, y up, z forward
                                    [-30, 0,  60, 1],
                                    [30, 0,  60, 1],
                                    [30, 0,  -60, 1]]).T
        self.vertices_W = self.T.dot(self.vertices_L)
        self.next_locations_by_steering = np.zeros((4,15), np.float32)
        self.next_locations_by_steering[3,:] = 1
        self.past_locations = []
        self.camera = camera
        self.camera.set_transform(y=Config.cam_height)
        self.displacement_vector = np.array([[0, 0, Config.displace_z, 1]]).T
        self.traffic_lights = traffic_lights
        self.attached_traffic_light = None

        self.init_kinematic_vars()
        self.init_reneder_options(play)

        self.set_transform(x=Config.vehicle_x, z=Config.vehicle_z)

    def init_kinematic_vars(self):
        # Kinematic network and variables as in:
        # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
        self.turn_angle = 0  # alpha in radians
        self.wheel_base = 120  # W length of car
        self.speed = 1  # d
        self.delta = 1  # unit of time here (unlike in World editor which is displacement.
        self.range_angle = (-0.785398, 0.785398)

    def init_reneder_options(self, play):
        if play:
            self.render_radius = 2
            self.render_thickness = -1
        else:
            self.render_radius = 15
            self.render_thickness = 5

        self.is_active = True
        self.render_next_locations_by_steering = False
        self.render_past_locations = False

        self.render_past_locations_thickness = 8
        self.render_past_locations_radius = 2

    def get_relevant_states(self):
        #It should contain only primitive datatypes, numpy arrays, lists of numpy arrays, no other User defined class
        states = {}
        states["cameraT"] = self.camera.T.copy()
        states["speed"] = self.speed
        states["T"] = self.T
        return states

    def render_next_locations_by_steering_func(self, image, C):
        if self.next_locations_by_steering.shape[1] > 1:
            x, y = C.project(self.next_locations_by_steering)
            for i in range(0, len(x)):
                thick = int(ceil(self.render_thickness / Config.r_ratio))
                radius = int(ceil(self.render_radius / Config.r_ratio))
                image = cv2.circle(image, (x[i], y[i]), radius , (0, 0, 255),thick)
        return image

    #@Override
    def render(self, image, C):
        image = super(Vehicle, self).render(image, C)
        if self.render_next_locations_by_steering == True:
            image = self.render_next_locations_by_steering_func(image,C)
        if self.render_past_locations == True:
            image = self.render_past_locations_func(image, C)

        return image

    def kinematic_model(self, z, x, yaw, delta):
        distance = self.speed * delta
        tan_steering = tan(self.turn_angle)
        beta_radians = (distance / self.wheel_base) * tan_steering
        r = self.wheel_base / tan_steering
        sinh_, sinhb_ = sin(yaw), sin(yaw + beta_radians)
        cosh_, coshb_ = cos(yaw), cos(yaw + beta_radians)
        z += -r * sinh_ + r * sinhb_
        x += r * cosh_ - r * coshb_
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
            min_x = int(Config.r_res[1] * 0.11)
            max_x = Config.r_res[1] - min_x
            x_pos = max(min_x, min(mouse[0], max_x))  # 45 degrees
            m_func = interp1d([min_x, max_x], [self.range_angle[0], self.range_angle[1]])
            self.turn_angle = m_func(x_pos)

    def append_past_location(self, past_location):
        if len(self.past_locations) > Config.num_past_poses:
            self.past_locations.pop(0)

        if len(past_location) == 3: # if it is received as a tuple
            self.past_locations.append([past_location[0],past_location[1],past_location[2],1])
        elif len(past_location) == 4: # if it is received as a transformation matrix
            x, y, z, roll, yaw, pitch = params_from_tansformation(past_location) #TODO maybe refactor this to be not depend directly on the outer package
            self.past_locations.append([x, y, z,1])

    # TODO rendering past locations at test time must contain data from true previous locations (cannot use vertices from train data(recirded))
    def render_past_locations_func(self, image, C):

        if len(self.past_locations) > 0:
            array_past_locations = np.array(self.past_locations[-1:-Config.num_past_poses:-Config.num_skip_poses]).T
            x, y = C.project(array_past_locations)
            for i in range(0, len(x)):
                thick = int(ceil(self.render_past_locations_thickness / Config.r_ratio))
                radius = int(self.render_past_locations_radius / Config.r_ratio)
                image = cv2.circle(image, (x[i], y[i]), radius, (128, 128, 128), thick)

        return image

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


    def set_camera_relative_transform(self,x=None, y=None, z=None, roll=None, yaw=None, pitch=None):
        current_params = params_from_tansformation(self.T)
        if x is None: x = current_params[0]
        if y is None: y = current_params[1]
        if z is None: z = current_params[2]
        if roll is None: roll = current_params[3]
        if yaw is None: yaw = current_params[4]
        if pitch is None: pitch = current_params[5]

        x_c, y_c, z_c, roll_c, yaw_c, pitch_c = self.camera.get_transform()
        rotated_displacement_vector = rot_y(yaw - pi).dot(self.displacement_vector)
        x += rotated_displacement_vector[0]
        z += rotated_displacement_vector[2]
        self.camera.set_transform(x, y_c, z, roll_c, yaw, pitch_c)

    def check_traffic_lights(self):

        x_c, y_c, z_c, roll_c, yaw_c, pitch_c = self.get_transform()
        vehicle_pos = np.array([[x_c, 0, z_c, 1]]).T

        # The following logic selects the closest traffic light and its color remains active (not gray) while the vehicle is still on the traffic light
        # closest_id = None
        # min_distance = 99999999.0
        for traffic_light in self.traffic_lights:
            distance = np.sqrt(np.sum(np.square(traffic_light.vertices_W - vehicle_pos), axis=0))
            min_distance_for_tl = distance.min()
            if min_distance_for_tl > 500:
                traffic_light.attached_to_vehicle = False
                self.attached_traffic_light = None
            else:
                if self.attached_traffic_light is  None:
                    traffic_light.attached_to_vehicle = True
                    self.attached_traffic_light = traffic_light
            pass

    def set_transform(self, x=None, y=None, z=None, roll=None, yaw=None, pitch=None):
        super(Vehicle, self).set_transform(x,y,z,roll,yaw, pitch)
        self.set_camera_relative_transform(x,y,z,roll,yaw, pitch)

        self.check_traffic_lights()

    def simulate(self, key_pressed, mouse):
        self.interpret_key(key_pressed)
        self.interpret_mouse(mouse)
        self.update_parameters()

    def compute_turn_angle(self, x,z):

        #I don't remember why I pus this condition here...
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
        desired = desired / (np.linalg.norm(desired)+ 1e-8)
        delta_angle = np.arccos(heading.dot(desired))
        cross = np.cross(heading, desired)

        plane_normal = np.array([0,-1,0])

        if plane_normal.dot(cross) > 0:
            delta_angle = -delta_angle

        next_turn_angle = math.atan(delta_angle * self.wheel_base/self.speed) #the desired turn angle that would take us from current yaw to desired yaw in an instant
        next_turn_angle = max(self.range_angle[0], min(next_turn_angle* 2, self.range_angle[1]))  # clamp the desired turn angle since that is not possible
        #I don't fucking know why it has to be multiplied by 2 but it works
        self.turn_angle = next_turn_angle

    def compute_speed(self, x, z):
        x_c, y_c, z_c, roll_c, yaw_c, pitch_c = self.get_transform()

        current = np.array([x_c, 0, z_c])
        desired = np.array([x, 0, z])
        difference = desired - current
        magnitude = np.linalg.norm(difference)
        print (self.speed, magnitude)
        interp_obj = interp1d([Config.min_waypoint_distance, Config.max_waypoint_distance], [0, Config.max_speed], fill_value="extrapolate")
        self.speed = min(interp_obj(magnitude), Config.max_speed)


    def simulate_given_waypoints(self, waypoints):
        """
        This method should not modify the speed, it will update the position and the orientation, given the desired location and orientation
        """
        x,z =waypoints[0][Config.test_waypoint_idx_speed],waypoints[2][Config.test_waypoint_idx_speed]
        self.compute_speed(x,z)
        x,z =waypoints[0][Config.test_waypoint_idx_steer],waypoints[2][Config.test_waypoint_idx_steer]
        self.compute_turn_angle(x,z)
        self.update_parameters()

