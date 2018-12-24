import cv2
import numpy as np
from simulator.util.Vehicle import Vehicle
from simulator.util.Camera import Camera
from simulator.util.LaneMarking import LaneMarking
from network.models.SimpleConv import ChauffeurNet
from simulator.util.Path import Path
import torch
from simulator.UI.GUI import GUI
from simulator.UI.GUI import EventBag
from simulator.UI.DataGeneration import Renderer
from network.train import Config

class Simulator(GUI):

    def __init__(self, event_bag_path = "", network_path = "", world_path=""):
        super(Simulator, self).__init__("Simulator", world_path=world_path)
        self.camera.set_transform(y=-1500)
        self.vehicle = Vehicle(self.camera)
        self.vehicle.set_transform(x = 100)
        self.world.actors.append(self.vehicle)

        #event_bag represents the data that the user generated. Has values such as the key pressed, and mouse
        self.event_bag = EventBag(event_bag_path, record=False)
        self.in_res = (72, 96)
        self.network_path = network_path


    #TODO refactor this because it is very similar to pre_simulate() from DataGeneration.py
    def get_path(self):
        all_states = []
        for i in range(len(self.event_bag)):
            key,x,y = self.event_bag.next_event()
            self.vehicle.simulate(key, (x,y))
            all_states.append(
                [self.vehicle.T.copy(), self.camera.C.copy(), self.vehicle.next_locations_by_steering.copy(),
                 self.vehicle.vertices_W.copy(), self.vehicle.turn_angle])
            # all_states.append([self.vehicle.T.copy(), self.camera.C.copy(), self.vehicle.next_locations.copy(), self.vehicle.vertices_W.copy(), self.vehicle.turn_angle])

        #resetting objects
        self.vehicle.set_transform(x=100, y=0,z=0,roll=0,yaw=0, pitch=0)
        self.camera.set_transform(x = 0, y=-1500, z = 0, roll=0, yaw=0, pitch=-1.5708)
        self.vehicle.render_radius = 15
        self.vehicle.render_thickness = 5
        self.vehicle.speed = 0
        self.event_bag.reset()

        path = Path(all_states)
        return path

    #@Override
    def run(self):
        #TODO Jesus refactor things because it's getting shittttyyy
        config = Config()
        config.horizon = 8
        model = ChauffeurNet(config)
        model.load_state_dict(torch.load(self.network_path))
        model.eval()
        model = model.to(config.device)

        path = self.get_path()

        image_test_nn = np.zeros((self.in_res[0], self.in_res[1], 3), np.uint8)

        with torch.no_grad():
            for i in range(len(self.event_bag)):
                key, x, y = self.event_bag.next_event()

                input_planes = Renderer.render_inputs_on_separate_planes(self.world,self.vehicle,path,i,self.in_res)
                input_planes_concatenated = Renderer.prepare_images(input_planes, self.in_res, debug = False)
                input_planes_concatenated = input_planes_concatenated[np.newaxis]
                input_planes_concatenated = torch.from_numpy(input_planes_concatenated.astype(np.float32) / 255.0 - 0.5).to(config.device)

                steering_pred, waypoints_pred = model(input_planes_concatenated)
                waypoints_coords = model.process_waypoints(waypoints_pred)




                # self.vehicle.simulate(key, None)
                self.vehicle.simulate(key, (x,y))

                image_test_nn = self.world.render(image=image_test_nn, C=self.camera)
                for i in range(len(waypoints_coords)):
                    image_test_nn = cv2.circle(image_test_nn,(waypoints_coords[i,1],waypoints_coords[i,0]),radius=1,color = (0,0+i*30,0),thickness=0)
                image_test_nn_resized = cv2.resize(image_test_nn,(640,480))
                cv2.imshow("Simulator", image_test_nn_resized)
                # cv2.imshow("input", input_to_network)
                cv2.waitKey(1)

    # @Override
    def interpret_key(self):
        key = self.pressed_key
        if key == 27:
            self.running = False

if __name__ =="__main__":
    simulator = Simulator(event_bag_path="../../data/recording.h5", network_path="../../data/ChauffeurNet.pt",
                          world_path="../../data/world.h5" )
    simulator.run()