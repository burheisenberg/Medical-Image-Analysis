import torch
from torch.utils.tensorboard import SummaryWriter
from model import ConvVAE
from data_loader import get_data_loaders
from utils import conv_vae_loss
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize parameters
batch_size = 32
learning_rate = 1e-3
latent_size = 20
epochs = 50
log_dir = 'logs'

# Get data loaders
train_loader, test_loader = get_data_loaders(batch_size=batch_size)

# Initialize the ConvVAE model and optimizer
model = ConvVAE(in_channels=1, out_channels=3, latent_size=latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_vae(model, train_loader, optimizer, num_epochs=5, log_dir='logs'):
    writer = SummaryWriter(log_dir)
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
    writer.close()

# Train the VAE model
train_vae(model, train_loader, optimizer, num_epochs=epochs, log_dir=log_dir)

# Optionally, save the model
torch.save(model.state_dict(), 'vae_model.pth')
