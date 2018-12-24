import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
from math import *
import matplotlib.pyplot as plt

"""
I define here the model, the dataset format and the training procedure for this specific model,
as these are tightly coupled
"""

class FeatureExtractor(nn.Module):

    def conv_block(self, in_channels, out_channels):
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding= 1)
        batc1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        batc2 = nn.BatchNorm2d(out_channels)
        relu2 = nn.ReLU(inplace=True)
        pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        block = nn.Sequential(conv1,batc1,relu1,conv2,batc2,relu2,pool2)
        return block

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.block1 = self.conv_block(6,16)
        self.block2 = self.conv_block(16,32)
        self.block3 = self.conv_block(32,64)

    def forward(self, x):
        L1 = self.block1(x)
        L2 = self.block2(L1)
        L3 = self.block3(L2)
        return [L1,L2,L3]

class Upsampling(nn.Module):

    def deconv_block(self, in_channels, out_channels):
        conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        batc1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1)
        #TODO change the parameters in convTranspose2D such thatwe obtain the doubled output after applying convtranspose and conv with padding 1
        batc2 = nn.BatchNorm2d(out_channels)
        relu2 = nn.ReLU(inplace=True)
        block = nn.Sequential(conv1,batc1,relu1,conv2,batc2,relu2)
        return block

    def __init__(self):
        super(Upsampling, self).__init__()

        self.block1 = self.deconv_block(64,32)
        self.block2 = self.deconv_block(32,16)
        self.block3 = self.deconv_block(16,16)

        return

    def forward(self, x):
        L1,L2,L3 = x[0],x[1],x[2]
        UP2 = self.block1(L3) + L2
        UP3 = self.block2(UP2) + L1
        UP4 = self.block3(UP3)
        return UP4

class SteeringPredictor(nn.Module):

    def __init__(self, hidden_size = 64 * 18 * 24):
        super(SteeringPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.drop1 = nn.Dropout(p=0.1, inplace=True)
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        return x

class WaypointHeatmap(nn.Module):

    def conv_block(self, in_channels, out_channels):
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1)
        batc1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.ReLU(inplace=True)
        block = nn.Sequential(conv1,batc1,relu1)
        return block

    def __init__(self):
        super(WaypointHeatmap, self).__init__()

        self.conv1 = self.conv_block(16,1)
        self.activation = nn.Softmax(dim=-1) # we want to do a spatial softmax

    def forward(self, x):
        x = self.conv1(x)
        x_before = x.data.cpu().numpy()
        x = self.activation(x)
        x_after = x.data.cpu().numpy()
        if False:
            fig, (ax1,ax2) = plt.subplots(1, 2)
            image_plot1 = ax1.imshow(np.squeeze(x_before[0, 0, ...]))
            image_plot2 = ax2.imshow(np.squeeze(x_after[0, 0, ...]))
            plt.colorbar(image_plot1, ax = ax1)
            plt.colorbar(image_plot2, ax = ax2)
            plt.show()

        return x


class AgentRNN(nn.Module):
    """
        Simple RNN as defined in https://pytorch.org/docs/stable/nn.html
        But instead of vectors we use convolutions
    """
    def __init__(self, config):
        super(AgentRNN, self).__init__()

        self.config             = config
        self.i2h                = nn.Conv2d(in_channels=1 , out_channels=16, kernel_size=3, padding=1) #waypoint     to hidden   required shape
        self.rel_i2h            = nn.ReLU(inplace=True)
        self.h2h                = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) #hidded       to hidden   required shape
        self.rel_h2h            = nn.ReLU(inplace=True)
        self.tan                = nn.Tanh()
        self.waypoint_predictor = WaypointHeatmap()
        self.upsampler         = Upsampling()


    def forward(self, x):
        # TODO initialize x_t0 to be the first waypoint produced by a convolution. x_t0 must not be the raw featuremaps
        x = self.upsampler(x)
        waypoint = self.waypoint_predictor(x) #I think the network should interpret the first waypoint as the current position, but I will add no loss to it
                                                #Since for the rest of the waypoints in range of horizon should be the future locations
        h_t = torch.zeros(waypoint.size(0), 16, waypoint.size(2), waypoint.size(3), dtype=torch.float32).to(self.config.device)
        future_waypoints = []
        for i in range(self.config.horizon):
            WihXt    = self.i2h(waypoint)
            WihXt    = self.rel_i2h(WihXt)

            WhhHt_1  = self.h2h(h_t)
            WhhHt_1  = self.rel_h2h(WhhHt_1)

            h_t      = self.tan(WihXt + WhhHt_1)
            waypoint = self.waypoint_predictor(h_t)
            future_waypoints.append(waypoint)

        future_waypoints = torch.stack(future_waypoints, dim=1)

        return future_waypoints

class ChauffeurNet(nn.Module):

    def __init__(self, config):
        super(ChauffeurNet, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.steering_predictor = SteeringPredictor()
        self.agent_rnn = AgentRNN(config)

        self.criterion_steering = nn.MSELoss(reduction='none')


    def forward(self, x):
        features = self.feature_extractor(x)
        steering = self.steering_predictor(features[2])
        waypoints = self.agent_rnn(features)

        return steering, waypoints

    def process_waypoints(self, waypoints_pred):
        # x_values, x_args =torch.max(waypoints_pred,dim=4, keepdim=True)
        # y_values_final, y_args_final = torch.max(x_values,dim=3, keepdim=True)
        # y_values, y_args = torch.max(waypoints_pred, dim=3, keepdim=True)
        # x_values_final, x_args_final = torch.max(y_values, dim=4, keepdim=True)
        # points = torch.stack([torch.squeeze(y_args_final), torch.squeeze(x_args_final)],dim=1)
        #Similar way, but I think its slower


        waypoints_pred = torch.squeeze(waypoints_pred,0)
        n = waypoints_pred.size(0)
        d = waypoints_pred.size(3)
        m = waypoints_pred.view(n, -1).argmax(1).view(-1, 1)
        indices = torch.cat((m // d, m % d), dim=1)
        indices = indices.cpu().numpy()
        # This way it gets y and x

        if False:
            for i in range(8):
                waypoints_pred = waypoints_pred.cpu()
                plt.clf()
                plt.imshow(np.squeeze(waypoints_pred[i, 0, ...]))
                plt.colorbar()
                plt.show()
        return indices

    def steering_weighted_loss(self, target, output):
        """
            Weight each example by an amount. If the error between gt and output is > 0.012 (0.70 degrees) then the penalty
            will be 5 otherwise the penalty is 0. This will force the network to learn better from hard examples and ignore
            allready learned examples.
        """
        diff = torch.abs(output - target)
        indices = ((diff > 0.0123599).type(torch.float32)) * 5.0
        weight = indices

        loss = self.criterion_steering(output, target)
        loss = loss * weight
        loss = loss.mean()
        return loss

    def waypoints_loss(self, future_penalty_maps,waypoints_pred):
        #some sort of focal loss. taken from
        #https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py
        #https://arxiv.org/pdf/1808.01244.pdf    eq (1)
        pos_inds = future_penalty_maps.eq(1)
        neg_inds = future_penalty_maps.lt(1)

        neg_weights = torch.pow(1 - future_penalty_maps[neg_inds], 4).float()
        loss = 0

        pos_pred = waypoints_pred[pos_inds]
        neg_pred = waypoints_pred[neg_inds]


        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

class EnumIndices:
    turn_angle_start_idx = 0
    future_points_start_idx = 1
    end_idx = int(future_points_start_idx + 2 * (40 / 5)) #LOOK IN Path class to know how many points there are

class DrivingDataset(Dataset):

    def __init__(self, hdf5_file, mode = "read", in_res = (144,192), num_channels = 6):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
        """
        range = (-0.785398, 0.785398)
        bins = 45
        self.in_res = in_res
        self.num_channels = 6
        self.mode = mode

        self.ratio = 3.333333

        if mode == "read":
            self.file = h5py.File(hdf5_file,"r", driver='core')
            self.dset_data = self.file['data']
            self.dset_target = self.file['labels']

            hist, bin_edges = np.histogram(self.dset_target[:,EnumIndices.turn_angle_start_idx], bins=bins, range=range, density=True)
            hist = hist / np.sum(hist)
            self.weighting_histogram = (1 / (hist + 0.01)).astype(np.float32)
            self.bin_edges = bin_edges
        elif mode == "write":
            self.file = h5py.File(hdf5_file, "w")
            self.dset_data = self.file.create_dataset("data", (0, self.num_channels, self.in_res[0], self.in_res[1]), dtype=np.uint8,
                                                         maxshape=(None, self.num_channels, self.in_res[0], self.in_res[1]),
                                                         chunks=(1, self.num_channels, self.in_res[0], self.in_res[1]))
            self.dset_labels = self.file.create_dataset("labels", (0, EnumIndices.end_idx), dtype=np.float32, maxshape=(None,  EnumIndices.end_idx),
                                                           chunks=(1, EnumIndices.end_idx))
            self.write_idx = 0

    def __len__(self):
        return self.dset_data.shape[0]

    def future_penalty_map(self, points):

        radius = int(ceil(20 / self.ratio))
        sigma = 0.3333 * radius


        points = np.reshape(points,(-1,2))
        num_points = points.shape[0]
        future_poses = np.zeros((num_points, 1, self.in_res[0], self.in_res[1]))

        for i in range(num_points):
            x_i,y_i = int(points[i,0]), int(points[i,1])
            for col in range(x_i - radius, x_i + radius):
                for row in range(y_i - radius, y_i + radius):
                    centred_col = col - x_i
                    centred_row = row - y_i
                    future_poses[i, 0, row, col] = exp(-((centred_col ** 2 + centred_row ** 2)) / (2 * sigma ** 2))

        if False:
            # fig, (ax1) = plt.subplots(1, 1)
            # for i in range(8):
            #     ax1.clear()
            #     image_plot1 = ax1.imshow(np.squeeze(future_poses[i, 0, ...]))
            #     plt.colorbar(image_plot1, ax = ax1)
            #     plt.show()
            for i in range(8):
                plt.imshow(np.squeeze(future_poses[i, 0, ...]))
                plt.show()
        return future_poses

    def __getitem__(self, idx):
        if self.mode != "read":
            raise ValueError ("Dataset opened with write mode")
        data = self.dset_data[idx,...].astype(np.float32) / 255.0 - 0.5
        steering = self.dset_target[idx,[EnumIndices.turn_angle_start_idx]]
        steering_bin = max(np.array([0]), min(np.digitize(steering, self.bin_edges) - 1, np.array([len(self.weighting_histogram) -1])))
        #DONT FORGET TO ADD [] when indexing...   [len(self.weighting_histogram) -1]    tensors need the same shape
        steering_weight = np.array(self.weighting_histogram[steering_bin])

        future_penalty_maps = self.future_penalty_map(self.dset_target[idx,EnumIndices.future_points_start_idx:EnumIndices.end_idx])

        sample = {'data': data, 'steering': steering, 'steering_weight':steering_weight, "future_penalty_maps": future_penalty_maps}
        return sample

    def append(self, images_concatenated, labels={}):
        if self.mode != "write":
            raise ValueError ("Dataset opened with read mode")

        turn_angle = np.array([labels["current_turn_angle"]])
        future_points = np.array(labels["points"]).astype(np.float32)
        future_points = future_points.reshape((-1,))
        stacked_labels = np.hstack((turn_angle,future_points))

        self.dset_data.resize((self.write_idx + 1, self.num_channels, self.in_res[0], self.in_res[1]))
        self.dset_labels.resize((self.write_idx + 1,  EnumIndices.end_idx))
        self.dset_data[self.write_idx, ...] = images_concatenated
        self.dset_labels[self.write_idx, ...] = stacked_labels
        self.write_idx +=1



def train_simple_conv(model, cfg, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sampled_batch in enumerate(train_loader):
        data = sampled_batch['data']
        steering_gt = sampled_batch['steering']
        steering_weight = sampled_batch['steering_weight']
        future_penalty_maps = sampled_batch['future_penalty_maps']

        data                = data.to(cfg.device)
        steering_gt         = steering_gt.to(cfg.device)
        steering_weight     = steering_weight.to(cfg.device)
        future_penalty_maps = future_penalty_maps.to(cfg.device)
        optimizer.zero_grad()
        steering_pred, waypoints_pred = model(data)

        loss_steering = model.steering_weighted_loss(steering_gt,steering_pred)
        loss_waypoints = model.waypoints_loss(future_penalty_maps,waypoints_pred)

        loss = loss_steering + loss_waypoints
        loss.backward()
        optimizer.step()
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            torch.save(model.state_dict(), os.path.join(cfg.checkpoints_path,"ChauffeurNet_{}_{}.pt".format(epoch,batch_idx)))

