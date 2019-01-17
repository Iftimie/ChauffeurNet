from .Actor import Actor
import numpy as np
from simulator.UI.GUI import GUI

class TrafficLight(Actor):

    def __init__(self, camera):
        """
        transform: 4x4 matrix to transform from local system to world system
        vertices_L: point locations expressed in local coordinate system in centimeters. vertices matrix will have shape
                4xN
        vertices_W: point locations expressed in world coordinate system
        """
        super().__init__()
        self.vertices_L = np.array([[0, 0, 0, 1],
                                    [100, 0, 0, 1]]).T
        self.vertices_W = self.T.dot(self.vertices_L)

        self.colours = {"green":(0,255,0),
                        "yellow":(255, 0,0),
                        "red":(0,255,255)}
        self.time_colours = {"green":330,
                            "yellow":120,
                            "red":330}

        self.camera = camera

        self.state = "green"
        self.c = (0,255,0)
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
        self.interpret_key(pressed_key)
        self.interpret_mouse(mouse)

        if self.crt_idx % self.time_colours[self.state] == 0:
            self.next_colour()

        self.crt_idx +=1
        print (self.crt_idx)


