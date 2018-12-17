import cv2
import numpy as np
import os
from util.World import World
from util.LaneMarking import LaneMarking
from util.TrafficLight import TrafficLight
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
WASD moves the selected actor
QE rotates the selected actor
Number keys: 1 LaneMarking
             2 TrafficLight
~ Delete selected actor
"""


class WorldEditor:

    def __init__(self):
        cv2.namedWindow("Editor")
        cv2.setMouseCallback("Editor", self.mouse_listener)

        self.world = World()
        if os.path.exists(self.world.save_path):
            self.world = self.world.load_world()
        else:
            self.world.actors.append(LaneMarking())
            self.world.actors.append(self.camera)

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
        image = np.zeros((480, 640, 3), np.uint8)

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
            self.delta = self.delta if self.delta>10 else 10
        if key == 9  or key == 49:
            self.select_actor(key)
        if key in [119, 100, 115, 97, 113, 101]:
            self.move_actor(key)
        if key in [49, 50, 51, 52]:
            self.add_actor(key)
        if key == 13:
            self.world.save_world(overwrite=True)
        if key == 96:
            self.delete_selected_actor()


        # if key == 112:
        #     self.camera.toggle_projection()

    def delete_selected_actor(self):
        if self.selected_actor != None:
            self.world.actors.remove(self.selected_actor)
            self.selected_index = 0

    def add_actor(self, key):
        if self.selected_actor != None:
            self.selected_actor.set_inactive()
        self.selected_index = len(self.world.actors) # we do that because we are adding one more actor at the end

        x, y, z, roll, yaw, pitch = self.camera.get_transform()
        new_actor = None
        if key == 49:
            new_actor = LaneMarking()
        if key == 50:
            new_actor = TrafficLight()

        new_actor.set_transform(x = x, y = 0, z = z)
        self.world.actors.append(new_actor)
        self.selected_actor = new_actor
        self.selected_actor.set_active()




    def move_actor(self, key):
        if self.selected_actor == None:return
        x, y, z, roll, yaw, pitch = self.selected_actor.get_transform()
        if key == 119:
            z +=self.delta
        if key == 100:
            x +=self.delta
        if key == 115:
            z -=self.delta
        if key == 97:
            x -=self.delta
        if key == 113:
            yaw -=self.delta / 10
        if key == 101:
            yaw +=self.delta / 10

        self.selected_actor.set_transform(x, y, z, roll, yaw, pitch)


    def select_actor(self, key):
        if self.selected_actor != None:
            self.selected_actor.set_inactive()
        if key == 9:
            self.selected_index+=1
            if self.selected_index > len(self.world.actors)-1: self.selected_index = 0
            self.selected_actor = self.world.actors[self.selected_index]
        if key == 49:
            self.selected_index-=1
            if self.selected_index < 0: self.selected_index = len(self.world.actors) -1
            self.selected_actor = self.world.actors[self.selected_index]
        self.selected_actor.set_active()

    def zoom(self, delta):
        x, y, z , roll, yaw, pitch = self.camera.get_transform()
        y+=delta
        self.camera.set_transform(x, y, z , roll, yaw, pitch)


if __name__ =="__main__":
    worldEditor = WorldEditor()
    worldEditor.edit()



