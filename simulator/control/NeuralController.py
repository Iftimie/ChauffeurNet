from simulator.control.Controller import Controller
from network.models.SimpleConv import ChauffeurNet
from network.train import Config as TrainingConfig
import torch

class NeuralController(Controller):
    """
    This controller should receive an
    """

    def __init__(self, vehicle, model_path):
        super(NeuralController, self).__init__(actor=vehicle)

        config = TrainingConfig()
        model = ChauffeurNet(config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model = model.to(config.device)

    def step(self, image):
