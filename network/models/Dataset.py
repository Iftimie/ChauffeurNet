import h5py
import numpy as np
from torch.utils.data import Dataset
from math import *
import matplotlib.pyplot as plt
from config import Config
from psutil import virtual_memory
import os

class EnumIndices:
    turn_angle_start_idx = 0
    future_points_start_idx = 1
    end_idx = int(future_points_start_idx + 2 * (40 / 5)) #LOOK IN Path class to know how many points there are

class DrivingDataset(Dataset):

    def __init__(self, hdf5_file, mode = "read", num_channels = 6):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
        """
        range = (-0.785398, 0.785398)
        bins = 45
        self.num_channels = 6
        self.mode = mode

        self.ratio = 3.333333

        if mode == "read":
            mem = virtual_memory()
            total_memory = mem.total / (1024*1024)
            file_size = os.path.getsize(hdf5_file) / (1024*1024)
            if not (total_memory - 8000 > file_size):
                self.file = h5py.File(hdf5_file,"r")
            else:
                self.file = h5py.File(hdf5_file, "r", driver='core')
            self.dset_data = self.file['data']
            self.dset_target = self.file['labels']

            hist, bin_edges = np.histogram(self.dset_target[:,EnumIndices.turn_angle_start_idx], bins=bins, range=range, density=True)
            hist = hist / np.sum(hist)
            self.weighting_histogram = (1 / (hist + 0.01)).astype(np.float32)
            self.bin_edges = bin_edges
        elif mode == "write":
            self.file = h5py.File(hdf5_file, "w")
            self.dset_data = self.file.create_dataset("data", (0, self.num_channels, Config.r_res[0], Config.r_res[1]), dtype=np.uint8,
                                                         maxshape=(None, self.num_channels, Config.r_res[0], Config.r_res[1]),
                                                         chunks=(1, self.num_channels, Config.r_res[0], Config.r_res[1]))
            self.dset_labels = self.file.create_dataset("labels", (0, EnumIndices.end_idx), dtype=np.float32, maxshape=(None,  EnumIndices.end_idx),
                                                           chunks=(1, EnumIndices.end_idx))
            self.write_idx = 0

    def __len__(self):
        return self.dset_data.shape[0]

    def future_penalty_map(self, points):

        #TODO points received in here should be in the full resolution. when downsampled to the network output, only then I should apply the fractional part regression

        scale_factor = Config.r_res[0] / Config.o_res[0]
        points /= scale_factor
        radius = int(ceil(20 / Config.o_ratio))
        sigma = 0.3333 * radius


        points = np.reshape(points,(-1,2))
        num_points = points.shape[0]
        future_poses = np.zeros((num_points, 1, Config.o_res[0], Config.o_res[1]))

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
        data = (self.dset_data[idx,...].astype(np.float32) - 128) / 128
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

        self.dset_data.resize((self.write_idx + 1, self.num_channels, Config.r_res[0], Config.r_res[1]))
        self.dset_labels.resize((self.write_idx + 1,  EnumIndices.end_idx))
        self.dset_data[self.write_idx, ...] = images_concatenated
        self.dset_labels[self.write_idx, ...] = stacked_labels
        self.write_idx +=1

