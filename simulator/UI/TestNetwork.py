import cv2
import numpy as np
from simulator.util.Vehicle import Vehicle
from simulator.util.Path import Path
import torch
from simulator.UI.GUI import GUI
from simulator.UI.Record import EventBag
from simulator.control.NeuralController import NeuralController
from config import Config


class Simulator(GUI):

    def __init__(self, event_bag_path = "", network_path = "", world_path=""):
        super(Simulator, self).__init__("Simulator", world_path=world_path)
        self.vehicle = Vehicle(self.camera)
        self.world.actors.append(self.vehicle)

        #event_bag represents the data that the user generated. Has values such as the key pressed, and mouse
        self.event_bag = EventBag(event_bag_path, record=False)
        self.path = Path(self.event_bag.list_states,debug=False)

        self.neural_controller = NeuralController(self.vehicle, self.world, network_path, self.path)

    def render_input_nn(self, path_idx, waypoints_2D):
        self.vehicle.c = (0,0,200)
        image_test_nn = np.zeros((Config.r_res[0], Config.r_res[1], 3), np.uint8)
        image_test_nn = self.path.render(image_test_nn, C=self.vehicle.camera, path_idx=path_idx, vehicle=self.vehicle)
        image_test_nn = self.world.render(image=image_test_nn, C=self.vehicle.camera, reset_image=False)
        image_test_nn = self.vehicle.render_past_locations_func(image=image_test_nn, C=self.vehicle.camera)
        for j in range(len(waypoints_2D)):
            image_test_nn = cv2.circle(image_test_nn, (waypoints_2D[j][1], waypoints_2D[j][0]), radius=2, color=(255, 0 + j * 30, 0), thickness=2)
        self.vehicle.c = (200, 200, 200)
        return image_test_nn

    #@Override
    def run(self):

        with torch.no_grad():
            path_idx = 0
            while path_idx < self.path.vertices_W.shape[1]:
                path_idx+=1

                waypoints_2D, path_idx = self.neural_controller.step(path_idx)
                image_test_nn = self.render_input_nn(path_idx, waypoints_2D)

                cv2.imshow("Simulator", image_test_nn)
                cv2.waitKey(1)

    # @Override
    def interpret_key(self):
        key = self.pressed_key
        if key == 27:
            self.running = False

if __name__ =="__main__":
    simulator = Simulator(event_bag_path="../../data/recorded_states.pkl", network_path="../../data/ChauffeurNet.pt",
                          world_path="../../data/world.h5" )
    simulator.run()