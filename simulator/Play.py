import cv2
import os
import numpy as np
import time
import h5py
from util.World import World
from util.Vehicle import Vehicle


def mouse_listener(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print (event)
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        # print (event)
        pass
    Simulator.mouse = (x, y)

class EnumMode:
    simulate = 0
    playback = 1

class Simulator:

    mouse = (320, 240)

    def __init__(self, mode = EnumMode.simulate):
        self.mode = mode
        cv2.namedWindow("Simulator")
        cv2.setMouseCallback("Simulator", mouse_listener)
        self.world = World()
        if os.path.exists(self.world.save_path):
            self.world = self.world.load_world()
            self.camera = self.world.get_camera_from_actors()
            self.camera.set_transform(y=-1500)
            self.vehicle = Vehicle(self.camera)
            self.vehicle.set_transform(x = 100)
            self.world.actors.append(self.vehicle)
        else:
            print ("No world available")

        self.time_step = 33

        if self.mode ==EnumMode.simulate:
            self.recording = []
        elif self.mode ==EnumMode.playback:
            self.iter = 0
            self.recording = self.load_recording()

    def run(self):
        image = np.zeros((480, 640, 3), np.uint8)

        while True:
            image = self.world.render(image=image, C=self.camera)
            cv2.imshow("Simulator", image)
            key = self.step()
            if key == 27:
                break

            if self.mode == EnumMode.playback:
                if self.iter == len(self.recording):break
                key = self.recording[self.iter,0]
                x = self.recording[self.iter,1]
                y = self.recording[self.iter,2]
                Simulator.mouse = (x,y)
                self.iter +=1

            self.vehicle.interpret_key(key)
            self.vehicle.interpret_mouse(Simulator.mouse)
            if key in [43, 45]:
                self.camera.interpret_key(key)
            self.vehicle.simulate()

            if self.mode == EnumMode.simulate:
                self.recording.append([key, Simulator.mouse[0], Simulator.mouse[1]])

        if self.mode == EnumMode.simulate:
            self.save_recording()
        print ("Game over")

    def save_recording(self):
        file = h5py.File("data/recording.h5", "w")
        all_recordings = np.array(self.recording)
        dset = file.create_dataset("recording", all_recordings.shape, dtype=np.int32)
        dset[...] = all_recordings
        file.close()

    def load_recording(self):
        file = h5py.File("data/recording.h5", "r")
        recording = file["recording"][...]
        file.close()
        return recording

    def step(self):
        prev_millis = int(round(time.time() * 1000))
        key = cv2.waitKey(self.time_step)
        curr_millis = int(round(time.time() * 1000))
        if curr_millis - prev_millis < self.time_step:
            seconds_to_sleep = (self.time_step - (curr_millis - prev_millis)) / 1000
            time.sleep(seconds_to_sleep)
        return key

if __name__ =="__main__":
    #simulator = Simulator(mode = EnumMode.simulate)
    simulator = Simulator(mode = EnumMode.playback)
    simulator.run()