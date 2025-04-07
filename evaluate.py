import torch
from model import ConvVAE
from data_loader import get_data_loaders
from utils import conv_vae_loss

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            recon_batch, mu, logvar = model(inputs)
            loss = conv_vae_loss(recon_batch, targets, mu, logvar)
            test_loss += loss.item()
    avg_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.6f}')
