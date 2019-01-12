import torch
import os

def train_simple_conv(model, cfg, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0

    print (len(optimizer.param_groups))
    for batch_idx, sampled_batch in enumerate(train_loader):

        optimizer.zero_grad()
        nn_outputs = model(sampled_batch['data'].to(cfg.device))
        loss = model.compute_loss(nn_outputs,sampled_batch, cfg)

        loss.backward()
        optimizer.step()

        total_loss += loss
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * cfg.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            torch.save(model.state_dict(), os.path.join(cfg.checkpoints_path,"ChauffeurNet_{}_{}.pt".format(epoch,batch_idx)))

    total_loss /= len(train_loader)
    cfg.scheduler.step(total_loss)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    del total_loss

