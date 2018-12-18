import cv2
import os
import numpy as np
import time

from util.World import World
from util.LaneMarking import LaneMarking
from util.Camera import Camera
from util.Vehicle import Vehicle


def mouse_listener(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print (event)
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        # print (event)
        pass
    Simulator.mouse = (x, y)

class Simulator:


    mouse = (320, 240)

    def __init__(self):
        cv2.namedWindow("Simulator")
        cv2.setMouseCallback("Simulator", mouse_listener)
        self.world = World()
        if os.path.exists(self.world.save_path):
            self.world = self.world.load_world()
            self.camera = self.world.get_camera_from_actors()
            self.vehicle = Vehicle(self.camera)
            self.vehicle.set_transform(x = 100)
            self.world.actors.append(self.vehicle)
        else:
            print ("No world available")

        self.time_step = 33

    def run(self):
        image = np.zeros((480, 640, 3), np.uint8)

        while True:
            image = self.world.render(image=image, C=self.camera)
            cv2.imshow("Simulator", image)
            key = self.step()

            self.vehicle.interpret_key(key)
            self.vehicle.interpret_mouse(Simulator.mouse)
            if key in [43, 45]:
                self.camera.interpret_key(key)
            self.vehicle.simulate()

    def step(self):
        prev_millis = int(round(time.time() * 1000))
        key = cv2.waitKey(self.time_step)
        curr_millis = int(round(time.time() * 1000))
        if curr_millis - prev_millis < self.time_step:
            seconds_to_sleep = (self.time_step - (curr_millis - prev_millis)) / 1000
            time.sleep(seconds_to_sleep)
        return key

if __name__ =="__main__":
    simulator = Simulator()
    simulator.run()