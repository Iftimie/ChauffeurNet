import cv2
import os
import numpy as np
from simulator.util.World import World
from simulator.util.Vehicle import Vehicle
from simulator.util.Camera import Camera
from simulator.util.LaneMarking import LaneMarking
from simulator.util.Path import Path
import atexit
from simulator.UI.GUI import EventBag
from network.models.SimpleConv import DrivingDataset

class Renderer:

    def __init__(self, world_path="", h5_path="", event_bag_path = ""):
        self.world = World(world_path = world_path)
        if not os.path.exists(self.world.save_path):
            raise ("No world available")
        self.world.load_world()
        self.camera = self.world.get_camera_from_actors()
        self.camera.set_transform(y = -1500)
        self.vehicle = Vehicle(self.camera, play =False)
        self.vehicle.set_transform(x = 100)
        self.world.actors.append(self.vehicle)

        self.h5_path = h5_path
        self.event_bag_path = event_bag_path
        self.event_bag = EventBag(self.event_bag_path, record=False)

        self.all_states = []

        self.in_res = (72, 96)

        self.dataset = DrivingDataset(self.h5_path, mode= "write")

        atexit.register(self.cleanup)

    def cleanup(self):
        self.dataset.file.close()

    def get_sin_noise(self):
        noise = np.sin(self.iter / 10) # * 5 (increase amplitude)
        return noise

    def add_noise_over_camera(self):
        cam_params = list(self.camera.get_transform())
        noise = self.get_sin_noise()
        #cam_params[0] += noise * 10 #x
        #cam_params[2] += noise * 10 #z
        cam_params[4] += noise / 20 #yaw
        self.camera.set_transform(*cam_params)

    def pre_simulate(self):
        for i in range(len(self.event_bag)):
            key, x, y = self.event_bag.next_event()
            self.vehicle.simulate(key, (x,y))

            self.all_states.append([self.vehicle.T.copy(), self.camera.C.copy(), self.vehicle.next_locations.copy(), self.vehicle.vertices_W.copy(), self.vehicle.turn_angle])

        self.event_bag.reset()
        self.path = Path(self.all_states)

    @staticmethod
    def render_on_separate_planes(world, vehicle, path):
        image_lanes = np.zeros((480, 640, 3), np.uint8)
        image_vehicle = np.zeros((480, 640, 3), np.uint8)
        image_path = np.zeros((480, 640, 3), np.uint8)

        for actor in world.actors:
            if type(actor) is Camera: continue
            if type(actor) is LaneMarking:
                image_lanes = actor.render(image_lanes, vehicle.camera)
        image_vehicle = vehicle.render(image_vehicle, vehicle.camera)
        image_path = path.render(image_path, vehicle.camera)
        # image_lanes = self.vehicle.render(image_lanes, self.camera)

        return [image_lanes, image_vehicle, image_path]

    @staticmethod
    def prepare_images(images, in_res):
        gray_images_resized = []
        for image in images:
            image_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_ = cv2.resize(image_, (in_res[1], in_res[0]))
            gray_images_resized.append(image_)
        image_concatenated = np.empty((3, in_res[0], in_res[1]), np.uint8)
        image_concatenated[0, ...] = gray_images_resized[0]
        image_concatenated[1, ...] = gray_images_resized[1]
        image_concatenated[2, ...] = gray_images_resized[2]
        return image_concatenated

    def render(self):

        self.pre_simulate()

        for i in range(len(self.all_states)):

            self.vehicle.T = self.all_states[i][0]
            self.camera.C =  self.all_states[i][1]
            self.vehicle.next_locations = self.all_states[i][2]
            self.vehicle.vertices_W = self.all_states[i][3]
            self.vehicle.turn_angle = self.all_states[i][4]

            image_planes = Renderer.render_on_separate_planes(self.world,self.vehicle, self.path)
            images_concatenated = Renderer.prepare_images(image_planes, self.in_res)
            self.dataset.append(images_concatenated,self.vehicle.turn_angle)

        print ("Rendering Done")

    def visualize(self):
        self.dataset.file.close()
        self.dataset = DrivingDataset(self.h5_path, mode = "read")

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            image_bgr = ((sample['data'] + 0.5) * 255.0).astype(np.uint8)
            label = sample['target']
            image_bgr = np.transpose(image_bgr, (1, 2, 0))
            print(label)
            cv2.imshow("image_bgr", image_bgr)
            cv2.waitKey(33)

if __name__ =="__main__":

    renderer = Renderer( world_path= "../data/world.h5", h5_path="../data/pytorch_data.h5", event_bag_path="../data/recording.h5")
    renderer.render()
    renderer.visualize()


