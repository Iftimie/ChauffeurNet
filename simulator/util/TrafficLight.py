from .Actor import Actor
import numpy as np
import cv2
from math import ceil
from config import Config

class TrafficLight(Actor):

    def __init__(self, obj_name):
        """
        from obj_name I extract the initial colour that the traffic light should have.
        should only be green and red

        """
        super().__init__()
        self.T = np.eye(4, dtype=np.float32)
        self.vertices_W = None
        self.line_pairs = None
        self.obj_name = obj_name
        self.colours = {"green":(0,255,0),
                        "yellow":(0, 255,255),
                        "red":(0,0,255),
                        "gray":(15,15,15)}
        self.time_colours = {"green":330,
                            "yellow":60,
                            "red":330}

        self.state = obj_name.split("_")[1]
        self.c = self.colours[self.state]
        self.crt_idx = 0

        self.attached_to_vehicle = False

    def next_colour(self):

        if self.crt_idx % self.time_colours[self.state] == 0:
            if self.state == "green":
                self.state = "yellow"
            elif self.state == "yellow":
                self.state = "red"
            elif self.state == "red":
                self.state = "green"
        self.crt_idx +=1

        if self.attached_to_vehicle:
            self.c = self.colours[self.state]
        else:
            self.c = self.colours["gray"]



    def simulate(self, pressed_key, mouse):
        pass


    def render(self, image, C, simulation_time=True):
        """
        :param image: image on which this actor will be renderd on
        :param C:     camera matrix
        :return:      image with this object renderd
        """

        if simulation_time:
            self.next_colour()


        if self.vertices_W.shape[1] > 1:
            x, y = C.project(self.vertices_W)
            pts = np.array([x, y]).T
            pts = pts.reshape((-1, 1, 2))
            for pair_idx in self.line_pairs:
                line_points = pts[pair_idx]
                thick = int(ceil(self.render_thickness / Config.r_ratio))
                image = cv2.polylines(image, [line_points], False, color=self.c,thickness= thick)

        return image
