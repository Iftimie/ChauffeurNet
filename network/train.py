import torch
import torch.optim as optim
import os
from time import gmtime, strftime
import shutil
torch.multiprocessing.set_sharing_strategy('file_system')


class Config:
    def __init__(self):
        torch.manual_seed(0)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.train_loader       = None
        self.model              = None
        self.optimizer          = None
        self.training_procedure = None
        self.experiment_name    = "whatever"

    def train(self, epoch):
        self.training_procedure(self.model, self, self.train_loader, self.optimizer, epoch)

    def initialize_experiment(self):
        if not os.path.exists("../experiments"):
            os.mkdir("../experiments")
        self.experiment_path = os.path.join("../experiments/",str(strftime("%Y-%m-%d_%H-%M-%S", gmtime())+"_"+self.experiment_name))
        os.mkdir(self.experiment_path)
        self.checkpoints_path = os.path.join(self.experiment_path,"checkpoints")
        os.mkdir(self.checkpoints_path)

        for path in self.paths_to_copy:
            destination = os.path.join(self.experiment_path,os.path.basename(path))
            if os.path.isdir(path):
                shutil.copytree(path, destination)
            else:
                shutil.copyfile(path, destination)


class ConfigSimpleConv(Config):
    def __init__(self, root_path = ".."):
        super().__init__()
        from network.models.SimpleConv import ChauffeurNet
        from network.models.Dataset import DrivingDataset
        from network.models.TrainUtil import train_simple_conv
        self.batch_size     = 64
        self.lr             = 0.005
        self.shuffle        = True
        self.epochs         = 100
        self.event_bag_path = os.path.join(root_path,"data/recorded_states.pkl")
        self.world_path     = os.path.join(root_path,"data/world.obj")
        self.log_interval   = 100
        self.experiment_name = "whatever"
        self.paths_to_copy = [os.path.join(root_path,"network/models/SimpleConv.py"),
                              os.path.join(root_path,"network/train.py"),
                              os.path.join(root_path,"simulator")]

        self.model = ChauffeurNet(config = self).to(self.device)
        self.train_loader = torch.utils.data.DataLoader(dataset = DrivingDataset(event_bag_path=self.event_bag_path,
                                                                                 world_path=self.world_path,
                                                                                 debug=False),
                                               batch_size=self.batch_size,
                                               shuffle=self.shuffle,
                                                num_workers=16)

        # self.model.load_state_dict(torch.load("../experiments/2018-12-21_21-38-57_whatever/checkpoints/ChauffeurNet_99_120.pt"))

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.training_procedure = train_simple_conv

        self.initialize_experiment()

def main():
    cfg = ConfigSimpleConv()
    for epoch in range(cfg.epochs):
        cfg.train(epoch)

if __name__ == '__main__':
    main()