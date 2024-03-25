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

from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler

from mdtv2 import MDTv2

#
# Config
#

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='data/LATENT_DATASET/LATENT_DATASET')
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--batch-size', type=int, default=16)
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

#
# Denoising Model
#

model: MDTv2 = MDTv2(depth=12, hidden_size=384, patch_size=2, num_heads=6, learn_sigma=False)
model = model.to(device)

if args.resume_path is not None:
    print(f'Resuming from {args.resume_path}')
    model.load_state_dict(torch.load(args.resume_path))

#
# Diffusion
#

scheduler = DDIMScheduler()

#
# Evaluation
#

def unconditional_sample(model, scheduler, num_samples):

    print("Generating...")

    model.eval()

    with torch.no_grad():

        x_t = torch.randn(num_samples, 4, 32, 32, device=device)

        scheduler.set_timesteps(50)

        for t in tqdm(scheduler.timesteps):
            # predict the noise residual
            with torch.no_grad():
                t = torch.tensor([t], device=device)
                noise_pred = model(x_t, t)

            # compute the previous noisy sample x_t -> x_t-1
            x_t = scheduler.step(noise_pred, t, x_t).prev_sample

        # Decode
        
        decoded = decode(x_t)

        # Save image

        name = f"generated_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        path = os.path.join(logdir, name)
        save_image(decoded, path, nrow=3, normalize=True, value_range=(-1, 1))

    model.train()
    

#
# Training
#

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train_loop(model, optimizer, dataloader):

    global_batch = 0
    global_step = 0

    # Training loop

    batch_size = args.batch_size

    for (mean, logvar) in dataloader:
        
        # Sample image from latent space

        mean, logvar = mean.to(device), logvar.to(device)
        x_0 = sample(mean, logvar)

        # Sample random noise & timesteps

        noise = torch.randn(x_0.shape, device=device)
        timesteps = torch.randint(0, 1000, (x_0.shape[0],), device=device)

        # Add noise

        x_t = scheduler.add_noise(x_0, noise, timesteps)

        # Forward pass: Predict the noise residuals

        predicted_noise = model(x_t, timesteps, enable_mask=True)

        # Calculate loss

        loss = F.mse_loss(predicted_noise, noise)

        # Backward pass

        loss.backward()

        # Update weights

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Update progress bar

        print(f"Step {global_step}, Loss: {loss.detach().item()}")

        global_batch += 1
        global_step += batch_size

        if global_batch % 100 == 0:
            print(f"Eval")

            # Save model
            torch.save(model.state_dict(), 'model.pth')
            torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

            # Evaluate
            unconditional_sample(model, scheduler, 1)


if not args.gen_only:
    train_loop(model, optimizer, dataloader)

unconditional_sample(model, scheduler, 1)


        
    


