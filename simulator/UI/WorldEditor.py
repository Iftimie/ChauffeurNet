import cv2
import numpy as np
import os
from simulator.util.World import World
from simulator.util.LaneMarking import LaneMarking
from simulator.util.CurvedLaneMarking import CurvedLaneMarking
from simulator.util.TrafficLight import TrafficLight
from simulator.util.Camera import Camera
from simulator.util.Vehicle import Vehicle
from simulator.UI.GUI import GUI

"""
Keys:
+ zoom in
- zoom out
P toggle orthographic/projective
Z increase delta (which moves the object)
X decrease delta (which moves the object)
TAB select next actor
ESC select previous actor
ENTER saves world
WASD moves the selected actor
QE rotates the selected actor
Number keys: 1 LaneMarking
             2 TrafficLight
             3 CurvedLaneMarking
             4 Vehicle
~ Delete selected actor
C select camera immediately
"""


class WorldEditor(GUI):

    def __init__(self, world_path=""):
        super(WorldEditor, self).__init__("Editor", world_path)

        self.selected_index = -1
        self.selected_actor = None
        self.camera.is_active = True

    #@Override
    def run(self):
        while True:
            super(WorldEditor, self).interact()
            if self.selected_actor is not None:
                self.selected_actor.interpret_key(self.pressed_key)

    def select_camera_immediately(self):
        if self.selected_actor != None:
            self.selected_actor.set_inactive()
        self.selected_actor = self.camera
        self.selected_actor.set_active()
        #keep the current index. do not change it

    def delete_selected_actor(self):
        if self.selected_actor != None and type(self.selected_actor) is not Camera:
            self.world.actors.remove(self.selected_actor)
            self.selected_index = 0

    def add_actor(self, key):
        if key == 49:
            new_actor = LaneMarking()
        if key == 50:
            new_actor = TrafficLight()
        if key ==51:
            new_actor = CurvedLaneMarking(arc_degree = 60, radius =300)
        if key == 52:
            new_actor = Vehicle()

        # Take the transformation params from the previous actor to ease the design process
        if self.selected_actor is not None and type(self.selected_actor) is not Camera:
            x, y, z, roll, yaw, pitch = self.selected_actor.get_transform()
            y = 0 # in case it is camera (which is up above (-1500 height))
            new_actor.set_transform(x, y, z, roll, yaw, pitch)
            self.selected_actor.set_inactive()
        else:
            x, y, z, roll, yaw, pitch = self.camera.get_transform()
            new_actor.set_transform(x=x, y=0, z=z)

        self.selected_index = len(self.world.actors)  # we do that because we are adding one more actor at the end
        self.world.actors.append(new_actor)
        self.selected_actor = new_actor
        self.selected_actor.set_active()

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

if __name__ =="__main__":
    worldEditor = WorldEditor("../data/world.h5")
    worldEditor.run()



