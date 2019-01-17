from simulator.control.Controller import Controller
from network.models.SimpleConv import ChauffeurNet
from network.models.Dataset import DrivingDataset
from network.train import Config as TrainingConfig
from simulator.UI.GUI import GUI
import numpy as np
import torch

class NeuralController(Controller):
    """
    This controller should receive a path idx based on which it will render the image, forward into net, and apply output to kinematic model of the car
    """

    def __init__(self, vehicle, world, model_path, recorded_path):
        """
        :param vehicle:         vehicle object to control
        :param world:           world necessary for rendering
        :param model_path:      path to neuralnet weights
        :param recorded_path:   Path object (contains previously recorded steps, necessary for GPS like directions)
        """
        super(NeuralController, self).__init__(actor=vehicle, world=world)

        self.config = TrainingConfig()
        model = ChauffeurNet(self.config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        self.model = model.to(self.config.device)
        self.path = recorded_path
        self.vehicle = self.registered_actor

    def step(self, path_idx):

        path_idx = self.path.get_point_idx_close_to_car(self.vehicle, path_idx)
        nn_input = self.render_neural_input(path_idx)

        nn_outputs = self.model(nn_input)
        waypoints_2D = self.model.process_waypoints(nn_outputs["waypoints"])
        waypoints_3D = []
        for waypoint in waypoints_2D:
            # mouse on world needs order x, y but the model returns y and x
            y, x = waypoint[0], waypoint[1]
            waypoints_3D.append(GUI.mouse_on_world((x, y), self.vehicle.camera))
        ones = np.ones((1, len(waypoints_3D)))
        waypoints_3D = np.squeeze(np.array(waypoints_3D)).T
        waypoints_3D = np.vstack((waypoints_3D, ones))

        self.vehicle.simulate_given_waypoints(waypoints_3D)

        return waypoints_2D, path_idx

    def render_neural_input(self, path_idx):
        input_planes = DrivingDataset.render_inputs_on_separate_planes(self.world, self.vehicle, self.path, path_idx)
        input_planes_concatenated = DrivingDataset.prepare_images(input_planes, debug=False)
        input_planes_concatenated = input_planes_concatenated[np.newaxis]
        input_planes_concatenated = torch.from_numpy((input_planes_concatenated.astype(np.float32) - 128) / 128).to(self.config.device)

        return input_planes_concatenated
