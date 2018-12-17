import cv2
import numpy as np

from util.World import World
from util.LaneMarking import LaneMarking
from util.Camera import Camera

"""
Keys:
+ zoom in
- zoom out
P toggle orthographic/projective
Z increase delta (which moves the object)
X decrease delta (which moves the object)
TAB select next actor
SHIFT select previous actor
"""


class WorldEditor:

    def __init__(self):
        cv2.namedWindow("Editor")
        cv2.setMouseCallback("Editor", self.mouse_listener)

        self.world = World()
        self.world.actors.append(LaneMarking())
        self.camera = Camera()
        self.delta = 10
        self.selected_index = -1
        self.selected_actor = None


    def mouse_listener(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print (event)
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            print (event)
            pass

    def edit(self):
        image = np.zeros((480, 640), np.uint8)

        while True:
            image = self.world.render(image=image, C=self.camera)
            cv2.imshow("Editor", image)
            key = cv2.waitKey(33)
            self.interpret_key(key)

    def interpret_key(self, key):
        if key != -1:
            print ("Pressed key ", key)
        if key == 43:
            self.zoom(delta = 100)
        if key == 45:
            self.zoom(delta= -100)
        if key == 122:
            self.delta +=10
        if key == 120:
            self.delta -=10
        if key == 9  or key == 49:
            self.select_actor(key)


        # if key == 112:
        #     self.camera.toggle_projection()

    def select_actor(self, key):
        if self.selected_actor != None:
            self.selected_actor.c = (255,)
        if key == 9:
            self.selected_index+=1
            if self.selected_index > len(self.world.actors)-1: self.selected_index = 0
            self.selected_actor = self.world.actors[self.selected_index]
        if key == 49:
            self.selected_index-=1
            if self.selected_index < 0: self.selected_index = len(self.world.actors) -1
            self.selected_actor = self.world.actors[self.selected_index]
        self.selected_actor.c = (155,)

    def zoom(self, delta):
        x, y, z , roll, yaw, pitch = self.camera.get_transform()
        y+=delta
        self.camera.set_transform(x, y, z , roll, yaw, pitch)


if __name__ =="__main__":
    worldEditor = WorldEditor()
    worldEditor.edit()



