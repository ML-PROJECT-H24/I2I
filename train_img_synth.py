import argparse
import torch
import PIL 
import os
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn

import latent_dataset
from torchvision.utils import save_image
import torch.nn.functional as F

from adan import Adan
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers.training_utils import compute_snr

from mdtv2 import MDTv2

#
# Config
#

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='data/LATENT_DATASET/LATENT_DATASET')
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--weight-decay', type=float, default=0)
argparser.add_argument('--mask-ratio', type=float, default=0.3)
argparser.add_argument('--batch-size', type=int, default=16)
argparser.add_argument('--epochs', type=int, default=256)
argparser.add_argument('--steps_per_epoch', type=int, default=10000)
argparser.add_argument('--resume-path', type=str, default=None)
argparser.add_argument('--gen-only', action='store_true', default=False)
argparser.add_argument('--logdir', type=str, default='logs')

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logdir = os.path.join(args.logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(logdir, exist_ok=True)

#
# Dataset
#

dataset = latent_dataset.LatentImageDataset(args.data_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

#
# Autoencoder
#

def init_vae(device):
    # https://huggingface.co/stabilityai/sd-vae-ft-mse
    model: AutoencoderKL = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device)
    model = model.eval()
    model.train = False
    for param in model.parameters():
        param.requires_grad = False
    return model

vae = init_vae(device)
scale_factor=0.18215 # scale_factor follows DiT and stable diffusion.

@torch.no_grad()
def encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
    posterior = vae.encode(x, return_dict=False)[0].parameters
    return torch.chunk(posterior, 2, dim=1)    

@torch.no_grad()
def sample(mean: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.FloatTensor:
    std = torch.exp(0.5 * logvar)
    z = torch.randn_like(mean)
    z = mean + z * std
    return z * scale_factor

@torch.no_grad()
def decode(z) -> torch.Tensor:
    x = vae.decode(z / scale_factor, return_dict=False)[0]
    # x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return x

def save_sample(x, path):
    save_image(x, path, nrow=3, normalize=True, value_range=(-1, 1))

#
# Denoising Model
#

model: MDTv2 = MDTv2(depth=12, hidden_size=384, patch_size=2, num_heads=6, learn_sigma=False, mask_ratio=args.mask_ratio)
model = model.to(device)

if args.resume_path is not None:
    print(f'Resuming from {args.resume_path}')
    model.load_state_dict(torch.load(args.resume_path))

#
# Diffusion
#

scheduler = DDIMScheduler()

#
# Sampling
#

def gen_img(model, scheduler):
    model.eval()

    with torch.no_grad():

        x_t = torch.randn(1, 4, 32, 32, device=device)

        scheduler.set_timesteps(50)

        for t in tqdm(scheduler.timesteps, desc="Sampling"):
            # predict the noise residual
            with torch.no_grad():
                tt = torch.tensor([t], device=device)
                noise_pred = model(x_t, tt)

            # compute the previous noisy sample x_t -> x_t-1
            x_t = scheduler.step(noise_pred, t, x_t).prev_sample

        # Decode
        
        decoded = decode(x_t)

    model.train()

    return decoded    

#
# Training
#

optimizer = Adan(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, max_grad_norm=1.0, fused=True)

def train_loop(model, optimizer, dataloader):

    batch_size = args.batch_size

    steps_per_epoch = args.steps_per_epoch
    batches_per_epoch = steps_per_epoch // batch_size
    epochs = args.epochs

    print(f"Steps per epoch: {steps_per_epoch}, Epochs: {epochs}")

    for i in range(epochs):
        # Run for batches_per_epoch batches

        pbar = tqdm(desc=f"Epoch {i}", total=steps_per_epoch)

        total_loss = 0

        for i in range(batches_per_epoch):
            # Get batch
    
            mean, logvar = next(iter(dataloader))
            mean, logvar = mean.to(device), logvar.to(device)

            # Sample image from latent space

            x_0 = sample(mean, logvar)

            # Sample random noise & timesteps

            noise = torch.randn(x_0.shape, device=device)
            timesteps = torch.randint(0, 1000, (x_0.shape[0],), device=device)

            # Add noise

            x_t = scheduler.add_noise(x_0, noise, timesteps)

            # Forward pass: Predict the noise residuals

            predicted_noise = model(x_t, timesteps, enable_mask=True)

            # Min-SNR: https://arxiv.org/pdf/2303.09556.pdf

            snr = compute_snr(scheduler, timesteps)

            base_weight = (torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)

            if scheduler.config.prediction_type == "v_prediction":
                # Velocity objective needs to be floored to an SNR weight of one.
                mse_loss_weights = base_weight + 1
            else:
                # Epsilon and sample both use the same loss weights.
                mse_loss_weights = base_weight

            mse_loss_weights[snr == 0] = 1.0

            # Calculate loss

            loss = F.mse_loss(predicted_noise, noise, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

            # Backward pass

            loss.backward()

            # Update weights

            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar

            total_loss += loss.detach().item()
            normalized_loss = total_loss / (i + 1)

            pbar.update(batch_size)
            pbar.set_postfix({"Loss": normalized_loss})
        
        # Save model
        try:
            torch.save(model.state_dict(), 'model.pth')
        except Exception as e:
            print(f"Error saving model: {e}")

        try:
            # Evaluate
            gen = gen_img(model, scheduler)

            name = f"generated_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            path = os.path.join(logdir, name)
            save_sample(gen, path)
        except Exception as e:
            print(f"Error generating sample: {e}")

train_loop(model, optimizer, dataloader)


        
    


