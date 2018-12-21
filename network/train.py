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
        self.experiment_path = os.path.join("../experiments/",strftime("%Y-%m-%d %H:%M:%S", gmtime()+"_"+self.experiment_name))
        os.mkdir(self.experiment_path)
        self.iterations_path = os.path.join(self.experiment_path)
        os.mkdir(self.iterations_path)

        shutil.copytree('../network', self.experiment_path)
        shutil.copytree('../simulator', self.experiment_path)


class ConfigSimpleConv(Config):
    def __init__(self):
        super().__init__()
        from models.SimpleConv import DrivingDataset, ChauffeurNet, train
        self.batch_size   = 128
        self.lr           = 0.0001
        self.shuffle      = True
        self.epochs       = 100
        self.dataset      = "../simulator/data/pytorch_data.h5"
        self.log_interval = 10
        self.experiment_name = "whatever"


        self.train_loader = torch.utils.data.DataLoader(dataset = DrivingDataset(self.dataset),
                                               batch_size=self.batch_size,
                                               shuffle=self.shuffle)
        self.model = ChauffeurNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.training_procedure = train

def main():
    cfg = ConfigSimpleConv()
    for epoch in range(cfg.epochs):
        cfg.train(epoch)
        # test(args, model, device, test_loader)

if __name__ == '__main__':
    main()