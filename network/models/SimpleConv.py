import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet,BasicBlock,model_urls
import torch.utils.model_zoo as model_zoo
from config import Config

"""
I define here the model, the dataset format and the training procedure for this specific model,
as these are tightly coupled
"""

class FeatureExtractor(ResNet):

    def __init__(self, imagenet_trained= False):
        super(FeatureExtractor, self).__init__(BasicBlock, [2, 2, 2, 2])

        #Resnet needs to reinitialize the firt convolution layer from 3 channels (RGB) to 6 channels (rendering planes)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if imagenet_trained:
            self.load_my_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        if Config.scale_factor == 2:
            x = self.layer1(x)
            return x
        elif Config.scale_factor == 4:
            x = self.layer1(x)
            x = self.layer2(x)
            return x
        elif Config.scale_factor == 8:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
        # x = self.layer4(x)

        return x

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                if name == "conv1.weight":
                    copy1 = param.data.clone()
                    copy2 = param.data.clone()
                    param = torch.cat([copy1, copy2], 1)
                else:
                    # backwards compatibility for serialized parameters
                    param = param.data
            own_state[name].copy_(param)

class SteeringPredictor(nn.Module):

    def __init__(self, hidden_size = Config.features_num_channels * Config.o_res[0] * Config.o_res[1]):
        super(SteeringPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.drop1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        return x

#defined new class that looks just like steering predictor
SpeedPredictor = SteeringPredictor


class WaypointHeatmap(nn.Module):

    def conv_block(self, in_channels, out_channels):
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1)
        batc1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.ReLU()
        block = nn.Sequential(conv1,batc1,relu1)
        return block

    def __init__(self):
        super(WaypointHeatmap, self).__init__()

        self.conv1 = self.conv_block(Config.rnn_num_channels,1)
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
        self.f2h                = nn.Conv2d(in_channels=Config.features_num_channels  , out_channels=Config.rnn_num_channels, kernel_size=3, padding=1) #feature      to hidden   required shape
        self.rel_f2h            = nn.ReLU()
        self.f2i                = nn.Conv2d(in_channels=Config.features_num_channels  , out_channels=Config.rnn_num_channels, kernel_size=3, padding=1) #feature      to input    required shape
        self.rel_f2i            = nn.ReLU()

        self.i2h                = nn.Conv2d(in_channels=Config.rnn_num_channels , out_channels=Config.rnn_num_channels, kernel_size=3, padding=1) #waypoint     to hidden   required shape
        self.rel_i2h            = nn.ReLU()
        self.h2h                = nn.Conv2d(in_channels=Config.rnn_num_channels, out_channels=Config.rnn_num_channels, kernel_size=3, padding=1) #hidded       to hidden   required shape
        self.rel_h2h            = nn.ReLU()
        self.tan                = nn.Tanh()
        self.waypoint_predictor = WaypointHeatmap()


    def forward(self, x):
        h_t = self.f2h(x)
        h_t = self.rel_f2h(h_t)

        x_t = self.f2i(x)
        x_t = self.rel_f2i(x_t)

        future_waypoints = []
        for i in range(Config.horizon_future):
            WihXt    = self.i2h(x_t)
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

        self.feature_extractor = FeatureExtractor(imagenet_trained=True)
        if "steering" in Config.nn_outputs:
            self.steering_predictor = SteeringPredictor()
            self.criterion_steering = nn.MSELoss(reduction='none')
        if "waypoints" in Config.nn_outputs:
            self.agent_rnn = AgentRNN(config)
        if "speed" in Config.nn_outputs:
            self.speed_predictor = SpeedPredictor()
            self.criterion_speed = nn.MSELoss(reduction='sum')



    def forward(self, x):
        features = self.feature_extractor(x)

        nn_outputs = {}
        if "steering" in Config.nn_outputs:
            nn_outputs["steering"] = self.steering_predictor(features)
        if "waypoints" in Config.nn_outputs:
            nn_outputs["waypoints"] = self.agent_rnn(features)
        if "speed" in Config.nn_outputs:
            nn_outputs["speed"] = self.speed_predictor(features)

        return nn_outputs

    def process_waypoints(self, waypoints_pred):

        waypoints_pred = torch.squeeze(waypoints_pred,0)
        n = waypoints_pred.size(0)
        d = waypoints_pred.size(3)
        m = waypoints_pred.view(n, -1).argmax(1).view(-1, 1)
        indices = torch.cat((m // d, m % d), dim=1)
        indices = indices.cpu().numpy()
        indices *= int(Config.scale_factor)
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

    def compute_loss(self, nn_outputs, sampled_batch, cfg):
        steering_gt = sampled_batch['steering'].to(cfg.device)
        speed_gt = sampled_batch['speed'].to(cfg.device)
        future_penalty_maps = sampled_batch['future_penalty_maps'].to(cfg.device)


        loss = 0
        if "steering" in nn_outputs:
            loss += self.steering_weighted_loss(steering_gt, nn_outputs["steering"])
        if "waypoints" in nn_outputs:
            loss += self.waypoints_loss(future_penalty_maps, nn_outputs["waypoints"])
        if "speed" in nn_outputs:
            loss += self.criterion_speed(speed_gt, nn_outputs["speed"]) / cfg.batch_size

        return loss

