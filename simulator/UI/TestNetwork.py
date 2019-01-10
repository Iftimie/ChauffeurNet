import cv2
import numpy as np
from simulator.util.Vehicle import Vehicle
from network.models.SimpleConv import ChauffeurNet
from simulator.util.Path import Path
import torch
from simulator.UI.GUI import GUI
from simulator.UI.Record import EventBag
from network.train import Config as TrainingConfig
from network.models.Dataset import DrivingDataset
from config import Config


class Simulator(GUI):

    def __init__(self, event_bag_path = "", network_path = "", world_path=""):
        super(Simulator, self).__init__("Simulator", world_path=world_path)
        self.vehicle = Vehicle(self.camera)
        self.world.actors.append(self.vehicle)

        #event_bag represents the data that the user generated. Has values such as the key pressed, and mouse
        self.event_bag = EventBag(event_bag_path, record=False)
        self.path = Path(self.event_bag.list_states,debug=False)
        self.network_path = network_path


    #@Override
    def run(self):
        #TODO Jesus refactor things because it's getting shittttyyy
        config = TrainingConfig()
        model = ChauffeurNet(config)
        model.load_state_dict(torch.load(self.network_path,map_location=config.device))
        model.eval()
        model = model.to(config.device)


        image_test_nn = np.zeros((Config.r_res[0], Config.r_res[1], 3), np.uint8)

        with torch.no_grad():
            path_idx = 0
            while path_idx < self.path.vertices_W.shape[1]:
                path_idx+=1
                state = self.event_bag.next_event()
                key,mouse = state["pressed_key"], state["mouse"]
                self.camera = state["vehicle"]["camera"]
                self.world.camera = state["vehicle"]["camera"]

                input_planes = DrivingDataset.render_inputs_on_separate_planes(self.world,self.vehicle,self.path,path_idx)
                input_planes_concatenated = DrivingDataset.prepare_images(input_planes, debug = False)
                copy_input_planes_concatenated = input_planes_concatenated.copy()
                input_planes_concatenated = input_planes_concatenated[np.newaxis]
                input_planes_concatenated = torch.from_numpy((input_planes_concatenated.astype(np.float32) - 128) / 128).to(config.device)

                nn_outputs = model(input_planes_concatenated)
                waypoints_2D = model.process_waypoints(nn_outputs["waypoints"])
                waypoints_3D = []
                for waypoint in waypoints_2D:
                    #mouse on world needs order x, y but the model returns y and x
                    y, x = waypoint[0], waypoint[1]
                    waypoints_3D.append(self.mouse_on_world((x, y), self.vehicle.camera))
                ones = np.ones((1, len(waypoints_3D)))
                waypoints_3D = np.squeeze(np.array(waypoints_3D)).T
                waypoints_3D = np.vstack((waypoints_3D,ones ))
                proj_waypoints_3D = self.vehicle.camera.project(waypoints_3D)


                # self.vehicle.simulate(key, mouse)
                self.vehicle.interpret_key(key)
                #TODO simulate given speed from network or from waypoint
                if "speed" in Config.nn_outputs:
                    pass
                    # print (nn_outputs["speed"] * Config.normalizing_speed)
                # self.vehicle.simulate_given_waypoints(x=waypoints_3D[0][Config.test_waypoint_idx], z=waypoints_3D[2][Config.test_waypoint_idx], yaw=None, mouse= None)
                self.vehicle.simulate_given_waypoints(waypoints_3D, yaw=None, mouse= None)

                self.vehicle.c = (0,0,200)
                image_test_nn.fill(0)
                image_test_nn, path_idx = self.path.render(image_test_nn, C=self.vehicle.camera, path_idx=path_idx, vehicle=self.vehicle, mode="test")
                image_test_nn = self.world.render(image=image_test_nn, C=self.vehicle.camera, reset_image=False)
                self.vehicle.c = (200,200,200)
                for j in range(len(waypoints_2D)):
                    image_test_nn = cv2.circle(image_test_nn,(waypoints_2D[j,1],waypoints_2D[j,0]),radius=1,color = (0,0+j*30,0),thickness=1)
                    image_test_nn = cv2.circle(image_test_nn, (proj_waypoints_3D[0][j], proj_waypoints_3D[1][j]), radius=2,color=(255, 0 + j * 30, 0), thickness=2)



                cv2.imshow("Simulator", image_test_nn)
                # for j in range(copy_input_planes_concatenated.shape[0]):
                #     cv2.imshow("input" + str(j), copy_input_planes_concatenated[j,...])
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