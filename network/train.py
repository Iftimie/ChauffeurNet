import torch
import torch.optim as optim
import os
from time import gmtime, strftime
import shutil

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
    def __init__(self):
        super().__init__()
        from models.SimpleConv import DrivingDataset, ChauffeurNet, train_simple_conv
        self.batch_size   = 128
        self.lr           = 0.0001
        self.shuffle      = True
        self.epochs       = 100
        self.dataset      = "../simulator/data/pytorch_data.h5"
        self.log_interval = 10
        self.experiment_name = "whatever"
        self.paths_to_copy = ["../network/models/SimpleConv.py","../network/train.py","../simulator"]


        self.train_loader = torch.utils.data.DataLoader(dataset = DrivingDataset(self.dataset),
                                               batch_size=self.batch_size,
                                               shuffle=self.shuffle)
        self.model = ChauffeurNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.training_procedure = train_simple_conv

        self.initialize_experiment()

def main():
    cfg = ConfigSimpleConv()
    for epoch in range(cfg.epochs):
        cfg.train(epoch)
        # test(args, model, device, test_loader)

if __name__ == '__main__':
    main()