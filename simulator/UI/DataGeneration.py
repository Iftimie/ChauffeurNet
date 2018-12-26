import cv2
import os
import numpy as np
from simulator.util.World import World
from simulator.util.Vehicle import Vehicle
from simulator.util.Camera import Camera
from simulator.util.LaneMarking import LaneMarking
from simulator.util.Path import Path
import atexit
from simulator.UI.Record import EventBag
from network.models.Dataset import DrivingDataset
from config import Config
import matplotlib.pyplot as plt

class Renderer:

    def __init__(self, world_path="", h5_path="", event_bag_path = "", overwrite = False,  debug= False):
        self.debug = debug

        self.world = World(world_path = world_path)
        self.world.load_world()

        self.camera = self.world.get_camera_from_actors()
        self.vehicle = Vehicle(self.camera, play =False)
        self.world.actors.append(self.vehicle)

        self.event_bag = EventBag(event_bag_path, record=False)
        self.path = Path(self.event_bag.list_states)

        atexit.register(self.cleanup)

        self.overwrite = overwrite
        self.h5_path = h5_path
        if overwrite == True or not os.path.exists(self.h5_path):
            self.dataset = DrivingDataset(self.h5_path, mode= "write")
        else:
            self.dataset = None
            print ("Dataset allready exists, not overwriting")
            return

    def cleanup(self):
        if self.dataset !=None:
            self.dataset.file.close()

    def add_noise_over_camera(self):
        def get_sin_noise():
            noise = np.sin(self.iter / 10)  # * 5 (increase amplitude)
            return noise
        cam_params = list(self.camera.get_transform())
        noise = get_sin_noise()
        #cam_params[0] += noise * 10 #x
        #cam_params[2] += noise * 10 #z
        cam_params[4] += noise / 20 #yaw
        self.camera.set_transform(*cam_params)

    @staticmethod
    def render_inputs_on_separate_planes(world, vehicle, path, path_idx):
        image_lanes             = np.zeros((Config.r_res[0],Config.r_res[1], 3), np.uint8)
        image_vehicle           = np.zeros((Config.r_res[0],Config.r_res[1], 3), np.uint8)
        image_path              = np.zeros((Config.r_res[0],Config.r_res[1], 3), np.uint8)
        image_agent_past_poses  = np.zeros((Config.r_res[0],Config.r_res[1], 3), np.uint8)

        for actor in world.actors:
            if type(actor) is Camera: continue
            if type(actor) is LaneMarking:
                image_lanes = actor.render(image_lanes, vehicle.camera)
        image_vehicle = vehicle.render(image_vehicle, vehicle.camera)
        image_path = path.render(image_path, vehicle.camera, path_idx)
        image_agent_past_poses = vehicle.render_past_locations_func(image_agent_past_poses, vehicle.camera)
        # image_lanes = self.vehicle.render(image_lanes, self.camera)

        input_planes = {"image_lanes":image_lanes,
                         "image_vehicle": image_vehicle,
                         "image_path": image_path,
                         "image_agent_past_poses":image_agent_past_poses}
        return input_planes

    @staticmethod
    def prepare_images(images, debug):

        image_lanes = images["image_lanes"]

        image_vehicle = images["image_vehicle"]
        image_vehicle = cv2.cvtColor(image_vehicle, cv2.COLOR_BGR2GRAY)

        image_path = images["image_path"]
        image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

        image_agent_past_poses = images["image_agent_past_poses"]
        image_agent_past_poses = cv2.cvtColor(image_agent_past_poses , cv2.COLOR_BGR2GRAY)



        image_concatenated = np.empty((6, Config.r_res[0], Config.r_res[1]), np.uint8)
        image_concatenated[0, ...] = image_lanes[...,0]
        image_concatenated[1, ...] = image_lanes[...,1]
        image_concatenated[2, ...] = image_lanes[...,2]
        image_concatenated[3, ...] = image_vehicle
        image_concatenated[4, ...] = image_path
        image_concatenated[5, ...] = image_agent_past_poses
        #TODO add the future pose here

        if debug:
            cv2.imshow("image1", image_lanes)
            cv2.imshow("image4", image_vehicle)
            cv2.imshow("image5", image_path)
            cv2.imshow("image6", image_agent_past_poses)
            cv2.waitKey(33)
        return image_concatenated

    def prepare_labels(self, path, path_idx):

        future_pose_states = {"points": [], "current_turn_angle":self.vehicle.turn_angle}
        # TODO I also have to add to the future pose states the angle prediction, or head orientation predicition
        for i in range(Config.horizon):
            point = path.project_future_poses(self.vehicle.camera, path_idx + i * Config.num_skip_poses)
            future_pose_states["points"].append(point)
        return future_pose_states

    def render(self):

        if self.overwrite == False:return

        for i in range(len(self.event_bag) - Config.num_future_poses):

            state = self.event_bag.next_event()
            self.vehicle.T = state["vehicle"]["T"]
            self.vehicle.append_past_location(self.vehicle.T) #TODO should refactor this so that any assignement to T updates every other dependent feature
            self.camera.C =  state["vehicle"]["camera"].C
            self.vehicle.next_locations_by_steering = state["vehicle"]["next_locations_by_steering"]
            self.vehicle.vertices_W = state["vehicle"]["vertices_W"]
            self.vehicle.turn_angle = state["vehicle"]["turn_angle"]

            self.path.apply_dropout(i)
            input_planes = Renderer.render_inputs_on_separate_planes(self.world,self.vehicle, self.path, i)
            input_planes_concatenated = Renderer.prepare_images(input_planes, self.debug)
            labels = self.prepare_labels(self.path, i)
            self.dataset.append(input_planes_concatenated,labels)

        print ("Rendering Done")

    def visualize(self):
        if self.dataset!=None:
            self.dataset.file.close()
        self.dataset = DrivingDataset(self.h5_path, mode = "read")

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            image_bgr = ((sample['data'] + 0.5) * 255.0).astype(np.uint8)
            label = sample['target']
            image_bgr = np.transpose(image_bgr, (1, 2, 0))
            print(label)
            cv2.imshow("image_bgr", image_bgr)
            cv2.waitKey(1)

if __name__ =="__main__":

    renderer = Renderer( world_path= "../../data/world.h5",
                         h5_path="../../data/pytorch_data.h5",
                         event_bag_path="../../data/recorded_states.pkl",
                         overwrite=True,
                         debug=True)
    # DONT FORGET TO CHANGE OVERWRITE
    renderer.render()
    # renderer.visualize()


