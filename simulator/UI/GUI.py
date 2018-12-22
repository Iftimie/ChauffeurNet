from simulator.util.World import World
from simulator.util.Camera import Camera
import cv2
import os
import numpy as np
import time
import h5py
import atexit

class GUI:

    mouse = (0,0)
    mouse_world = (0,0,0)
    @staticmethod
    def mouse_listener(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print (event)
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            print (event)
            pass
        GUI.mouse = (x,y)

    def mouse_on_world(self, mouse):
        """
        Don't use the GUI.mouse variable....or maybe use it???
        http://antongerdelan.net/opengl/raycasting.html
        Given the mouse click, find the 3D point on the ground plane
        """
        #I Don't fucking understand why I have to fucking substract mouse.x from 640. I just don't
        mouse_homogeneous = np.array([[640-mouse[0],mouse[1],1,1]]).T
        ray_eye = np.linalg.inv(self.camera.K).dot(mouse_homogeneous)
        ray_eye[2] = 1
        ray_eye[3] = 0
        ray_world = np.linalg.inv(self.camera.T).dot(ray_eye)
        ray_world = ray_world / np.linalg.norm(ray_world)
        ray_world = ray_world[:3, :]
        plane_normal = np.array([[0, -1, 0]]).T  # the plane normal is oriented towards the sky. objects up above have a - sign
        cam_params = self.camera.get_transform()
        O = np.array([[cam_params[0], cam_params[1], cam_params[2]]]).T

        t = -O.T.dot(plane_normal) / ray_world.T.dot(plane_normal)
        point_on_plane = O + ray_world * t
        # point_on_plane[0] *=-1

        return point_on_plane

    def __init__(self, window_name = "", world_path=""):

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
        self.display_image = np.zeros((480, 640, 3), np.uint8)
        self.window_name = window_name
        self.running = True
        self.time_step = 33

    def step(self):
        prev_millis = int(round(time.time() * 1000))
        key = cv2.waitKey(self.time_step)
        curr_millis = int(round(time.time() * 1000))
        if curr_millis - prev_millis < self.time_step:
            seconds_to_sleep = (self.time_step - (curr_millis - prev_millis)) / 1000
            time.sleep(seconds_to_sleep)
        return key

    def interact(self):
        """
        Will render the world, will listen for keyboard and mouse inputs
        """
        self.pressed_key = self.step()
        GUI.mouse_world = self.mouse_on_world(GUI.mouse)
        self.interpret_key()  # will call child class method
        self.display_image = self.world.render(image=self.display_image, C=self.camera)
        cv2.imshow(self.window_name, self.display_image)
        # self.world.simulate(self.pressed_key, GUI.mouse)

    def run(self):
        pass

    def interpret_key(self):
        pass

class EventBag:

    def __init__(self, file_path, record = True):
        self.record = record
        if record == True:
            self.file = h5py.File(file_path, "w")
            #TODO change the name from recording to events
            self.dset = self.file.create_dataset("recording", shape=(0,3), maxshape=(None,3),chunks=(1,3), dtype=np.int32)
        else:
            self.file = h5py.File(file_path, "r")
            self.dset = self.file['recording']

        self.crt_idx = 0
        atexit.register(self.cleanup)

    def append(self, events):
        if self.record == True:
            self.dset.resize((self.crt_idx + 1, 3))
            self.dset[self.crt_idx,...] = np.array(events)
            self.crt_idx +=1
        else:
            raise ValueError("EventBag opened as read mode")

    def __len__(self):
        return self.dset.shape[0]

    def next_event(self):
        if self.record == False:
            key = self.dset[self.crt_idx,0]
            x = self.dset[self.crt_idx,1]
            y = self.dset[self.crt_idx,2]
            self.crt_idx +=1
        else:
            raise ValueError("EventBag opened as write mode")
        return key,x,y

    def reset(self):
        self.crt_idx = 0

    def cleanup(self):
        self.file.close()
