import argparse
import torch
import PIL 
import os
import datetime
import numpy as np
from tqdm import tqdm

import latent_dataset
from torchvision.utils import save_image
import torch.nn.functional as F

from diffusers.models import AutoencoderKL
from diffusers import UNet2DModel
from diffusers import DDPMScheduler


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

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# Model
#

model = UNet2DModel(
    sample_size=32,  # the target image resolution
    in_channels=4, 
    out_channels=4, 
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 32, 64, 64),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D", 
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

mean, logvar = next(iter(dataloader))
mean, logvar = mean.to(device), logvar.to(device)

sampled = sample(mean, logvar)

print("Input shape:", sampled.shape)
print("Output shape:", model(sampled, timestep=0).sample.shape)

#
# DDPM
#

total_timesteps = 1000
noise_scheduler = DDPMScheduler(num_train_timesteps=total_timesteps)

#noise = torch.randn(sampled.shape, device=device)

# timesteps = torch.tensor([600], device=device)

# noisy_image = noise_scheduler.add_noise(sampled, noise, timesteps)

# decoded = decode(noisy_image)

# Display

# name = f"generated_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
# save_image(decoded, name, nrow=3, normalize=True, value_range=(-1, 1))

#
# Evaluation
#

def evaluate(epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=args.batch_size,
        generator=torch.manual_seed(args.seed),
    ).images

    for _, image in enumerate(images):
        decoded = decode(image)
        # Save image
        name = f"generated_{epoch}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        save_image(decoded, name, nrow=3, normalize=True, value_range=(-1, 1))

#
# Training
#

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train_loop(model, optimizer, dataloader):

    global_batch = 0
    global_step = 0

    # Training loop

    batch_size = args.batch_size

    pbar = tqdm(dataloader)

    for (mean, logvar) in pbar:
        
        # Sample image from latent space

        mean, logvar = mean.to(device), logvar.to(device)
        x_0 = sample(mean, logvar)

        # Sample random noise & timesteps

        noise = torch.randn(x_0.shape, device=device)
        timesteps = torch.randint(0, total_timesteps, (x_0.shape[0],), device=device)

        # Add noise

        x_t = noise_scheduler.add_noise(x_0, noise, timesteps)

        # Forward pass: Predict the noise residuals

        predicted_noise = model(x_t, timesteps, return_dict=False)[0]

        # Calculate loss

        loss = F.mse_loss(predicted_noise, noise)

        # Backward pass

        loss.backward()

        # Update weights

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Update progress bar

        logs = {"loss": loss.item(), "step": global_step}

        pbar.set_postfix(logs)

        global_batch += 1
        global_step += batch_size

        if global_batch % 1000 == 0:
            # Save model
            torch.save(model.state_dict(), 'model.pth')

            


train_loop(model, optimizer, dataloader)


        
    


