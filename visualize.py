import matplotlib.pyplot as plt
import torch

def visualize_results(model, data_loader, num_images=5):
    model.eval()
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        outputs, _, _ = model(inputs)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5))
    for i in range(num_images):
        axes[i, 0].imshow(targets[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(outputs[i].cpu().permute(1, 2, 0))
        axes[i, 1].set_title("Reconstructed")
    plt.show()
