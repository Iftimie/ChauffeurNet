import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os

class DrivingDataset(Dataset):

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
        """
        file = h5py.File(hdf5_file,"r")
        self.dset_data = file['data']
        self.dset_target = file['labels']
        #self.weighting_histogram

        range = (-0.785398, 0.785398)
        bins = 45
        # self.bins = np.arange(range[0], range[1], (range[1] - range[0]) / bins)
        # self.bins = np.hstack((self.bins, np.array([range[1]])))
        hist,bin_edges = np.histogram(self.dset_target[...],bins=bins, range=range, density=True)
        hist = hist / np.sum(hist)
        self.weighting_histogram = (1 / (hist+0.01)).astype(np.float32)
        self.bin_edges = bin_edges


    def __len__(self):
        return self.dset_data.shape[0]

    def __getitem__(self, idx):
        data = self.dset_data[idx,...].astype(np.float32)
        data /= 255.0
        data -= 0.5
        target = self.dset_target[idx,...]
        target_bin = np.digitize(target, self.bin_edges) - 1
        if target_bin >= len(self.weighting_histogram) :target_bin = np.array([len(self.weighting_histogram) -1])
        if target_bin <= 0 :target_bin = np.array([0])
        #DONT FORGET TO ADD [] when indexing...   [len(self.weighting_histogram) -1]    tensors need the same shape
        weight = np.array(self.weighting_histogram[target_bin])
        sample = {'data': data, 'target': target, 'weight':weight}
        return sample

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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    log_interval = 10
    criterion = nn.MSELoss(reduction='none')
    for batch_idx, sampled_batch in enumerate(train_loader):
        data = sampled_batch['data']
        target = sampled_batch['target']
        weight = sampled_batch['weight']
        data, target, weight = data.to(device), target.to(device), weight.to(device)
        optimizer.zero_grad()
        output = model(data)

        diff = torch.abs(output - target)
        indices = ((diff > 0.0123599).type(torch.float32)) * 5.0
        weight = indices

        loss = criterion(output, target)
        loss = loss * weight
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main():
    # Training settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)

    batch_size = 128
    lr = 0.0001
    shuffle = True
    epochs = 100
    #TODO will shuflle now, but when using LSTM, we will not
    dataset = DrivingDataset("../simulator/data/pytorch_data.h5")
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)

    model = ChauffeurNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)

    torch.save(model.state_dict(), "ChauffeurNet.pt")


if __name__ == '__main__':
    main()