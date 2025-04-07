# ConvVAE

Modular PyTorch implementation of a Convolutional Variational Autoencoder with TensorBoard support.

## Features

- TensorBoard logging
- MSE + KL divergence loss
- Image reconstruction visualization

## Usage

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train**

   ```bash
   python train.py
   ```

3. **Monitor**

   ```bash
   tensorboard --logdir=logs
   ```

4. **Evaluate**

   ```bash
   python evaluate.py
   ```

5. **Visualize**

   ```bash
   python visualize.py
   ```

## Structure

- `data_loader.py` – Dataset pipeline  
- `model.py` – ConvVAE architecture  
- `train.py` – Training loop  
- `evaluate.py` – Evaluation  
- `visualize.py` – Output visualizations  
- `utils.py` – Losses and helpers  

## Requirements

Python 3.8+, PyTorch, torchvision, matplotlib, tensorboard, tqdm

## License

MIT

