import torch
import os

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

