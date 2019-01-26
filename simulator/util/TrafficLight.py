from .Actor import Actor
import numpy as np
from simulator.UI.GUI import GUI
import cv2
from math import ceil
from config import Config

class TrafficLight(Actor):

    def __init__(self, camera):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        """
        super().__init__()
        self.vertices_L = np.array([[ 858.1092437,     0.,         -388.65546218, 1.0],
                                    [ 956.30252101,    0.,         -393.82352941, 1.0],
                                    [1028.65546218,    0.,         -393.82352941, 1.0],
                                    [1106.17647059,    0.,         -398.99159664, 1.0],
                                    [1163.02521008,    0.,         -435.16806723, 1.0],
                                    [1183.69747899,    0.,         -486.8487395 , 1.0]]).T
        self.T = np.eye(4, dtype=np.float32)
        self.vertices_W = self.vertices_L#self.T.dot(self.vertices_L)

        self.colours = {"green":(0,255,0),
                        "yellow":(0, 255,255),
                        "red":(0,0,255)}
        self.time_colours = {"green":330,
                            "yellow":60,
                            "red":330}

        self.camera = camera

        self.state = "green"
        self.c = self.colours[self.state]
        self.time = 165 # how many frames a light should stay in a colour
        self.crt_idx = 0

    def next_colour(self):
        if self.state == "green":
            self.c = self.colours["yellow"]
            self.state = "yellow"
        elif self.state == "yellow":
            self.c = self.colours["red"]
            self.state = "red"
        elif self.state == "red":
            self.c = self.colours["green"]
            self.state = "green"

    def set_active(self):
        print ("Available keys: N")

    def set_inactive(self):
        pass

    def interpret_key(self, key):
        if key == "N":
            pass

    def interpret_mouse(self, mouse):
        point = GUI.mouse_on_world(mouse, self.camera)

    def simulate(self, pressed_key, mouse):
        # self.interpret_key(pressed_key)
        # self.interpret_mouse(mouse)

        if self.crt_idx % self.time_colours[self.state] == 0:
            self.next_colour()

        self.crt_idx +=1

    def render(self, image, C):
        """
        :param image: image on which this actor will be renderd on
        :param C:     camera matrix
        :return:      image with this object renderd
        """
        if self.vertices_L.shape[1] > 1:
            x, y = C.project(self.vertices_L)
            pts = np.array([x, y]).T
            pts = pts.reshape((-1, 1, 2))
            thick = int(ceil(self.render_thickness / Config.r_ratio))
            image = cv2.polylines(image, [pts], False, color=self.c,thickness= thick)
        return image

