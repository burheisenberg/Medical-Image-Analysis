import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(input_folder, output_folder, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(input_folder, output_folder, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
