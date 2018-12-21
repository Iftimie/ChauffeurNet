from util.World import World
from util.Camera import Camera
import cv2
import os
import numpy as np

class GUI:

    mouse = (0,0)
    @staticmethod
    def mouse_listener(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print (event)
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            print (event)
            pass
        GUI.mouse = (x,y)

    def __init__(self, window_name = ""):

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, GUI.mouse_listener)

        self.world = World()
        if os.path.exists(self.world.save_path):
            self.world.load_world()
            self.camera = self.world.get_camera_from_actors()
        else:
            self.camera = Camera()
            self.world.actors.append(self.camera)

        self.pressed_key = None
        self.display_image = np.zeros((480, 640, 3), np.uint8)
        self.window_name = window_name

    def interact(self):
        """
        Will render the world, will listen for keyboard and mouse inputs
        """
        self.display_image = self.world.render(image=self.display_image, C=self.camera)
        cv2.imshow(self.window_name, self.display_image)
        self.pressed_key = cv2.waitKey(33)
        self.interpret_key() #will call child class method

    def run(self):
        pass

    def interpret_key(self):
        pass
