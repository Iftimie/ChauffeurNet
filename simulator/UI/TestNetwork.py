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
            all_states.append([self.vehicle.T.copy(), self.camera.C.copy(), self.vehicle.next_locations.copy(), self.vehicle.vertices_W.copy(), self.vehicle.turn_angle])

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
        model = ChauffeurNet()
        model.load_state_dict(torch.load(self.network_path))
        model.eval()

        path = self.get_path()

        image_test_nn = np.zeros((480, 640, 3), np.uint8)

        with torch.no_grad():
            for i in range(len(self.event_bag)):
                key, x, y = self.event_bag.next_event()

                image_lanes = np.zeros((480, 640, 3), np.uint8)
                image_vehicle = np.zeros((480, 640, 3), np.uint8)
                image_path = np.zeros((480, 640, 3), np.uint8)

                for actor in self.world.actors:
                    if type(actor) is Camera: continue
                    if type(actor) is LaneMarking:
                        image_lanes = actor.render(image_lanes, self.camera)
                image_vehicle = self.vehicle.render(image_vehicle, self.camera)
                image_path = path.render(image_path, self.camera, i)

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

                input_to_network = np.transpose(np.squeeze(image_concatenated), (1, 2, 0))

                image_concatenated = image_concatenated.astype(np.float32) / 255.0 - 0.5

                self.vehicle.turn_angle = model(torch.from_numpy(image_concatenated))
                self.vehicle.turn_angle = np.array(self.vehicle.turn_angle)[0,0]
                print (self.vehicle.turn_angle)
                self.vehicle.simulate(key, None)

                image_test_nn = self.world.render(image=image_test_nn, C=self.camera)
                cv2.imshow("Simulator", image_test_nn)
                cv2.imshow("input", input_to_network)
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