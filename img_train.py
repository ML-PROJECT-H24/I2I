import argparse
import torch
import PIL 
import os
import datetime
import numpy as np
from tqdm.auto import tqdm

import latent_dataset
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from adan import Adan
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers.training_utils import compute_snr

from mdtv2 import MDTv2

#
# Config
#

argparser = argparse.ArgumentParser()
argparser.add_argument('--data-dir', type=str, default='data/LATENT_DATASET/LATENT_DATASET')
argparser.add_argument('--num-classes', type=int, default=1000)
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--weight-decay', type=float, default=0)
argparser.add_argument('--mask-ratio', type=float, default=0.3)
argparser.add_argument('--batch-size', type=int, default=16)
argparser.add_argument('--epochs', type=int, default=512)
argparser.add_argument('--samples-per-epoch', type=int, default=None)
argparser.add_argument('--resume-path', type=str, default=None)
argparser.add_argument('--no-eval', action='store_true', default=False)
argparser.add_argument('--logdir', type=str, default='logs')

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logdir = os.path.join(args.logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(logdir, exist_ok=True)

#
# Dataset
#

print(f"Loading dataset from {args.data_dir}")

dataset = latent_dataset.LatentImageDataset(args.data_dir)

print(f"Loaded {len(dataset)} samples")

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

def dataloader_generator(dataloader):
    while True:
        yield from dataloader

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
    return x

def save_sample(x, path):
    save_image(x, path, normalize=True, value_range=(-1, 1))

#
# Denoising Model
#

num_classes = args.num_classes

model: MDTv2 = MDTv2(
    depth=12, 
    hidden_size=384, 
    patch_size=2, 
    num_heads=6, 
    num_classes=num_classes, 
    learn_sigma=False, 
    mask_ratio=args.mask_ratio,
    class_dropout_prob=0)

model = model.to(device)

if args.resume_path is not None:
    print(f'Resuming from {args.resume_path}')
    model.load_state_dict(torch.load(args.resume_path))

#
# Diffusion
#

num_train_timesteps = 1000
noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

#
# Sampling
#

def gen_samples(model, noise_scheduler, num_samples = 8):
    model.eval()

    with torch.no_grad():

        x_t = torch.randn(num_samples, 4, 32, 32, device=device)
        
        if num_classes == 2:
            half = num_samples // 2
            classes_rand = torch.cat([torch.zeros(half), torch.ones(num_samples - half)], dim=0).long().to(device)
        else:
            classes_rand = torch.randint(0, num_classes, (num_samples,), device=device)

        noise_scheduler.set_timesteps(50)

        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            # predict the noise residual
            with torch.no_grad():
                # Classifier free guidance
                tt = torch.tensor([t], device=device)
                noise_pred = model(x_t, tt, classes_rand, enable_mask=False)

            # compute the previous noisy sample x_t -> x_t-1
            x_t = noise_scheduler.step(noise_pred, t, x_t).prev_sample

    model.train()

    return x_t

#
# Training
#

accelerator = Accelerator(mixed_precision='fp16')

optimizer = Adan(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, max_grad_norm=1.0, fused=True)

def train_loop(model, optimizer, dataloader, noise_scheduler):
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    batch_size = args.batch_size
    steps_per_epoch = args.samples_per_epoch if args.samples_per_epoch is not None else len(dataloader) * batch_size
    batches_per_epoch = steps_per_epoch // batch_size
    
    epochs = args.epochs

    print(f"Samples per epoch: {steps_per_epoch}")
    print(f"Steps per epoch: {batches_per_epoch}")
    print(f"Total epochs: {epochs}")

    data_generator = dataloader_generator(dataloader)

    for epoch in range(epochs):
        # Run for batches_per_epoch batches

        pbar = tqdm(desc=f"Epoch {epoch}", total=steps_per_epoch)

        total_loss = 0

        for batch_idx in range(batches_per_epoch):

            # Get next batch

            mean, logvar, dict = next(data_generator)
            mean, logvar = mean.to(device), logvar.to(device)

            cond = dict['y'] % num_classes

            # Sample image from latent space

            x_0 = sample(mean, logvar)

            # Sample random noise & timesteps

            noise = torch.randn(x_0.shape, device=device)
            timesteps = torch.randint(0, num_train_timesteps, (x_0.shape[0],), device=device)

            # Add noise

            x_t = noise_scheduler.add_noise(x_0, noise, timesteps)

            # Forward pass: Predict the noise residuals

            predicted_noise = model(x_t, timesteps, cond, enable_mask=True)

            # Min-SNR: https://arxiv.org/pdf/2303.09556.pdf

            snr = compute_snr(noise_scheduler, timesteps)

            base_weight = (torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)

            mse_loss_weights = base_weight
            mse_loss_weights[snr == 0] = 1.0

            # Calculate loss

            loss = F.mse_loss(predicted_noise, noise, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

            # Backward pass

            accelerator.backward(loss)

            # Update weights
            
            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar

            total_loss += loss.detach().item()

            pbar.update(batch_size)
            pbar.set_postfix({"Loss": total_loss / (batch_idx + 1)})
        
        
        pbar.close()

        unwrapped_model = accelerator.unwrap_model(model)
        
        # Save model
        try:
            torch.save(unwrapped_model.state_dict(), os.path.join(logdir, f"model.pt"))
        except Exception as e:
            print(f"Error saving model: {e}")

        if not args.no_eval:
            try:
                # Evaluate
                x_0 = gen_samples(unwrapped_model, noise_scheduler)
                decoded = decode(x_0)

                name = f"generated_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                path = os.path.join(logdir, name)
                save_sample(decoded, path)
            except Exception as e:
                print(f"Error generating sample: {e}")

train_loop(model, optimizer, dataloader, noise_scheduler)


        
    


