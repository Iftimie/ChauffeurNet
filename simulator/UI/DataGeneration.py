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
import pickle

class Renderer:

    def __init__(self, world_path="", h5_path="", event_bag_path = "", overwrite = False, do_presimulate = False):
        self.world = World(world_path = world_path)
        if not os.path.exists(self.world.save_path):
            raise ("No world available")
        self.world.load_world()
        self.camera = self.world.get_camera_from_actors()
        self.camera.set_transform(y = -1200)
        self.vehicle = Vehicle(self.camera, play =False)
        self.vehicle.set_transform(x = 100)
        self.world.actors.append(self.vehicle)

        self.h5_path = h5_path
        self.event_bag_path = event_bag_path
        self.event_bag = EventBag(self.event_bag_path, record=False)

        self.in_res = (72*3, 96*3)


        self.all_states = []


        atexit.register(self.cleanup)

        self.overwrite = overwrite
        self.do_presimulate = do_presimulate
        if overwrite == True or not os.path.exists(self.h5_path):
            self.dataset = DrivingDataset(self.h5_path, mode= "write", in_res=self.in_res)
        else:
            self.dataset = None
            print ("Dataset allready exists, not overwriting")
            return


    def cleanup(self):
        if self.dataset !=None:
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
        if os.path.exists("../../data/tmp_all_states.pkl") and not self.do_presimulate:
            print ("Loading cached all states")
            self.all_states = pickle.load(open("../../data/tmp_all_states.pkl","rb"))
        else:
            print ("Creating all states")
            for i in range(len(self.event_bag)):
                key, x, y = self.event_bag.next_event()
                self.vehicle.simulate(key, (x,y))

                self.all_states.append([self.vehicle.T.copy(), self.camera.C.copy(), self.vehicle.next_locations_by_steering.copy(), self.vehicle.vertices_W.copy(), self.vehicle.turn_angle])
            pickle.dump(self.all_states,open("../../data/tmp_all_states.pkl","wb"))
        self.event_bag.reset()
        self.path = Path(self.all_states)

    @staticmethod
    def render_on_separate_planes(world, vehicle, path, path_idx):
        image_lanes = np.zeros((480, 640, 3), np.uint8)
        image_vehicle = np.zeros((480, 640, 3), np.uint8)
        image_path = np.zeros((480, 640, 3), np.uint8)
        image_agent_past_poses = np.zeros((480, 640, 3), np.uint8)

        for actor in world.actors:
            if type(actor) is Camera: continue
            if type(actor) is LaneMarking:
                image_lanes = actor.render(image_lanes, vehicle.camera)
        image_vehicle = vehicle.render(image_vehicle, vehicle.camera)
        image_path = path.render(image_path, vehicle.camera, path_idx)
        image_agent_past_poses = vehicle.render_past_locations_func(image_agent_past_poses, vehicle.camera)
        # image_lanes = self.vehicle.render(image_lanes, self.camera)

        return {"image_lanes":image_lanes,"image_vehicle": image_vehicle, "image_path": image_path, "image_agent_past_poses":image_agent_past_poses}

    @staticmethod
    def prepare_images(images, in_res):
        gray_images_resized = []

        image_lanes = images["image_lanes"]
        image_lanes = cv2.resize(image_lanes, (in_res[1], in_res[0]))

        image_vehicle = images["image_vehicle"]
        image_vehicle = cv2.cvtColor(image_vehicle, cv2.COLOR_BGR2GRAY)
        image_vehicle = cv2.resize(image_vehicle, (in_res[1], in_res[0]))

        image_path = images["image_path"]
        image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
        image_path = cv2.resize(image_path, (in_res[1], in_res[0]))

        image_agent_past_poses = images["image_agent_past_poses"]
        image_agent_past_poses = cv2.cvtColor(image_agent_past_poses , cv2.COLOR_BGR2GRAY)
        image_agent_past_poses = cv2.resize(image_agent_past_poses , (in_res[1], in_res[0]))

        # for image in images:
        #     image_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     image_ = cv2.resize(image_, (in_res[1], in_res[0]))
        #     gray_images_resized.append(image_)
        image_concatenated = np.empty((6, in_res[0], in_res[1]), np.uint8)
        image_concatenated[0, ...] = image_lanes[...,0]
        image_concatenated[1, ...] = image_lanes[...,1]
        image_concatenated[2, ...] = image_lanes[...,2]
        image_concatenated[3, ...] = image_vehicle
        image_concatenated[4, ...] = image_path
        image_concatenated[5, ...] = image_agent_past_poses

        cv2.imshow("image1", image_lanes)
        cv2.imshow("image4", image_vehicle)
        cv2.imshow("image5", image_path)
        cv2.imshow("image6", image_agent_past_poses)
        cv2.waitKey(33)
        return image_concatenated

    def render(self):

        if self.overwrite == False:return
        self.pre_simulate()

        for i in range(len(self.all_states)):

            self.vehicle.T = self.all_states[i][0]
            self.vehicle.append_past_location(self.vehicle.T) #TODO should refactor this so that any assignement to T updates every other dependent feature
            self.camera.C =  self.all_states[i][1]
            self.vehicle.next_locations = self.all_states[i][2]
            self.vehicle.vertices_W = self.all_states[i][3]
            self.vehicle.turn_angle = self.all_states[i][4]

            image_planes = Renderer.render_on_separate_planes(self.world,self.vehicle, self.path, i)
            images_concatenated = Renderer.prepare_images(image_planes, self.in_res)
            self.dataset.append(images_concatenated,self.vehicle.turn_angle)

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
                         event_bag_path="../../data/recording.h5",
                         overwrite=True,
                         do_presimulate=True)
    # DONT FORGET TO CHANGE OVERWRITE
    renderer.render()
    renderer.visualize()


