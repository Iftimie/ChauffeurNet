import cv2
import os
import numpy as np
import time
import h5py
from util.World import World
from util.Vehicle import Vehicle
from util.Camera import Camera
from util.LaneMarking import LaneMarking
from network.train import ChauffeurNet
from util.Path import Path
import torch


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
    test     = 2

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

    #TODO refactor this because it is very similar to pre_simulate() from DataGeneration.py
    def get_path(self, recordings):
        all_states = []
        for i in range(recordings.shape[0]):
            key = recordings[i, 0]
            mouse = (recordings[i, 1], recordings[i, 2])
            self.vehicle.interpret_key(key)
            self.vehicle.interpret_mouse(mouse)
            self.vehicle.simulate()
            all_states.append([self.vehicle.T.copy(), self.camera.C.copy(), self.vehicle.next_locations.copy(), self.vehicle.vertices_W.copy(), self.vehicle.turn_angle])
        self.vehicle.set_transform(x=100, y=0,z=0,roll=0,yaw=0, pitch=0)
        self.camera.set_transform(x = 0, y=-1500, z = 0, roll=0, yaw=0, pitch=-1.5708)
        path = Path(all_states)
        return path

    def test(self):
        model = ChauffeurNet()
        model.load_state_dict(torch.load("../network/ChauffeurNet.pt"))

        recordings = self.load_recording()
        path = self.get_path(recordings)
        self.in_res = (72, 96)

        image_test_nn = np.zeros((480, 640, 3), np.uint8)

        with torch.no_grad():
            for i in range(recordings.shape[0]):

                key = recordings[i, 0]

                image_lanes = np.zeros((480, 640, 3), np.uint8)
                image_vehicle = np.zeros((480, 640, 3), np.uint8)
                image_path = np.zeros((480, 640, 3), np.uint8)

                for actor in self.world.actors:
                    if type(actor) is Camera: continue
                    if type(actor) is LaneMarking:
                        image_lanes = actor.render(image_lanes, self.camera)
                image_vehicle = self.vehicle.render(image_vehicle, self.camera)
                image_path = path.render(image_path, self.camera)

                images = [image_lanes,image_vehicle,image_path]
                gray_images_resized = []
                for image in images:
                    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_ = cv2.resize(image_, (self.in_res[1], self.in_res[0]))
                    gray_images_resized.append(image_)
                image_concatenated = np.empty((1, 3, self.in_res[0], self.in_res[1]), np.uint8)
                image_concatenated[0,0, ...] = gray_images_resized[0]
                image_concatenated[0,1, ...] = gray_images_resized[1]
                image_concatenated[0,2, ...] = gray_images_resized[2]
                image_concatenated = image_concatenated.astype(np.float32) / 255.0

                self.vehicle.interpret_key(key)
                self.vehicle.turn_angle = model(torch.from_numpy(image_concatenated))
                self.vehicle.turn_angle = np.array(self.vehicle.turn_angle)[0,0]
                print (self.vehicle.turn_angle)
                self.vehicle.simulate()

                image_test_nn = self.world.render(image=image_test_nn, C=self.camera)
                cv2.imshow("Simulator", image_test_nn)
                cv2.waitKey(1)



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
    #simulator = Simulator(mode = EnumMode.playback)
    #simulator.run()
    simulator = Simulator(mode = EnumMode.test)
    simulator.test()