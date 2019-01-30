from simulator.util.World import World
from simulator.util.Camera import Camera
import cv2
import os
import numpy as np
import time
from config import Config

class GUI:

    mouse = (0,0, 0)
    mouse_world = (0,0,0)
    @staticmethod
    def mouse_listener(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            pass
        GUI.mouse = (x,y, event)

    @staticmethod
    def mouse_on_world( mouse, camera):
        """
        Don't use the GUI.mouse variable....or maybe use it???
        http://antongerdelan.net/opengl/raycasting.html
        Given the mouse click, find the 3D point on the ground plane
        """
        mouse_homogeneous = np.array([[mouse[0],mouse[1],1,1]]).T
        ray_eye = np.linalg.inv(camera.K).dot(mouse_homogeneous)

        ray_eye[2] = 1
        ray_eye[3] = 0
        ray_world = camera.T.dot(ray_eye)
        ray_world = ray_world / np.linalg.norm(ray_world)
        ray_world = ray_world[:3, :]
        plane_normal = np.array([[0, -1, 0]]).T  # the plane normal is oriented towards the sky. objects up above have a - sign
        cam_params = camera.get_transform()
        O = np.array([[cam_params[0], cam_params[1], cam_params[2]]]).T

        t = -O.T.dot(plane_normal) / ray_world.T.dot(plane_normal)
        point_on_plane = O + ray_world * t
        return point_on_plane

    def __init__(self, window_name = "", world_path=""):

        if not Config.linux_env:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, GUI.mouse_listener)

        self.world = World(world_path = world_path)
        if os.path.exists(self.world.save_path):
            self.world.load_world()
            self.camera = self.world.get_camera_from_actors()
            print ("World loaded")
        else:
            self.camera = Camera()
            self.world.actors.append(self.camera)
            print ("Warning, world not loaded, creating a new one")

        self.pressed_key = None
        self.display_image = np.zeros((Config.r_res[0], Config.r_res[1], 3), np.uint8)
        self.window_name = window_name
        self.running = True
        self.time_step = 33

    def step(self):
        if not Config.linux_env:
            prev_millis = int(round(time.time() * 1000))
            key = cv2.waitKey(self.time_step)
            curr_millis = int(round(time.time() * 1000))
            if curr_millis - prev_millis < self.time_step:
                seconds_to_sleep = (self.time_step - (curr_millis - prev_millis)) / 1000
                time.sleep(seconds_to_sleep)
        else:
            key=-1
        return key

    def interpretIO_and_render(self):
        """
        Will render the world, will listen for keyboard and mouse inputs
        """
        self.pressed_key = self.step()
        GUI.mouse_world = GUI.mouse_on_world(GUI.mouse, self.camera)
        self.interpret_key()  # will call child class method
        self.display_image = self.world.render(image=self.display_image, C=self.camera)
        cv2.imshow(self.window_name, self.display_image)
        # self.world.simulate(self.pressed_key, GUI.mouse)

    def run(self):
        pass

    def interpret_key(self):
        pass
