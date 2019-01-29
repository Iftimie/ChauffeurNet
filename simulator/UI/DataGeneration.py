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

    renderer = Renderer( world_path= "../../data/world.obj",
                         h5_path="../../data/pytorch_data.h5",
                         event_bag_path="../../data/recorded_states.pkl",
                         overwrite=True,
                         debug=True)
    # DONT FORGET TO CHANGE OVERWRITE
    renderer.render()
    # renderer.visualize()


