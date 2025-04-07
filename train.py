import torch
from torch.utils.tensorboard import SummaryWriter
from model import ConvVAE
from data_loader import get_data_loaders
from utils import conv_vae_loss

def train_vae(model, train_loader, optimizer, num_epochs=5, log_dir='logs'):
    writer = SummaryWriter(log_dir)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = conv_vae_loss(recon_batch, targets, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(inputs):.6f}')
        avg_loss = total_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        scheduler.step()
    writer.close()
