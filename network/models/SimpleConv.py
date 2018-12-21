import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset
import os

"""
I define here the model, the dataset format and the training procedure for this specific model,
as these are tightly coupled
"""

class ChauffeurNet(nn.Module):

    def conv_block(self, in_channels, out_channels):
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        batc1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        batc2 = nn.BatchNorm2d(out_channels)
        relu2 = nn.ReLU(inplace=True)
        pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        block = nn.Sequential(conv1,batc1,relu1,conv2,batc2,relu2,pool2)
        return block

    def __init__(self):
        super(ChauffeurNet, self).__init__()

        self.block1 = self.conv_block(3,16)
        self.block2 = self.conv_block(16,32)
        self.block3 = self.conv_block(32,64)

        self.drop1 = nn.Dropout(p=0.1, inplace=True)
        self.fc1 = nn.Linear(64*5*8, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, 64*5*8)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        return x

class DrivingDataset(Dataset):

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
        """
        file = h5py.File(hdf5_file,"r")
        self.dset_data = file['data']
        self.dset_target = file['labels']

        range = (-0.785398, 0.785398)
        bins = 45
        hist,bin_edges = np.histogram(self.dset_target[...],bins=bins, range=range, density=True)
        hist = hist / np.sum(hist)
        self.weighting_histogram = (1 / (hist+0.01)).astype(np.float32)
        self.bin_edges = bin_edges

    def __len__(self):
        return self.dset_data.shape[0]

    def __getitem__(self, idx):
        data = self.dset_data[idx,...].astype(np.float32) / 255.0 - 0.5
        target = self.dset_target[idx,...]
        target_bin = max(np.array([0]), min(np.digitize(target, self.bin_edges) - 1, np.array([len(self.weighting_histogram) -1])))
        #DONT FORGET TO ADD [] when indexing...   [len(self.weighting_histogram) -1]    tensors need the same shape
        weight = np.array(self.weighting_histogram[target_bin])
        sample = {'data': data, 'target': target, 'weight':weight}
        return sample

def step_weighting_loss(target, output, criterion):
    """
        Weight each example by an amount. If the error between gt and output is > 0.012 (0.70 degrees) then the penalty
        will be 5 otherwise the penalty is 0. This will force the network to learn better from hard examples and ignore
        allready learned examples.
    """
    diff = torch.abs(output - target)
    indices = ((diff > 0.0123599).type(torch.float32)) * 5.0
    weight = indices

    loss = criterion(output, target)
    loss = loss * weight
    loss = loss.mean()
    return loss

def train_simple_conv(model, cfg, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.MSELoss(reduction='none')
    for batch_idx, sampled_batch in enumerate(train_loader):
        data = sampled_batch['data']
        target = sampled_batch['target']
        weight = sampled_batch['weight']
        data, target, weight = data.to(cfg.device), target.to(cfg.device), weight.to(cfg.device)
        optimizer.zero_grad()
        output = model(data)

        loss = step_weighting_loss(target,output,criterion)

        loss.backward()
        optimizer.step()
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            torch.save(model.state_dict(), os.path.join(cfg.checkpoints_path,"ChauffeurNet_{}_{}.pt".format(epoch,batch_idx)))

