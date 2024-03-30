import argparse
import torch
import os
import sys
import datetime
import numpy as np
import functools
from tqdm.auto import tqdm

import latent_dataset
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from adan import Adan
from diffusers.models import AutoencoderKL
from gaussian_diffusion import *

from mdtv2 import MDTv2

#
# Config
#

argparser = argparse.ArgumentParser()
argparser.add_argument('--data-dir', type=str, default='data/LATENT_DATASET/LATENT_DATASET')
argparser.add_argument('--num-classes', type=int, default=2)
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--weight-decay', type=float, default=0)
argparser.add_argument('--mask-ratio', type=float, default=0.3)
argparser.add_argument('--batch-size', type=int, default=16)
argparser.add_argument('--steps-per-epoch', type=int, default=100)
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
    hidden_size=768, 
    patch_size=2, 
    num_heads=12, 
    num_classes=num_classes, 
    learn_sigma=True, 
    mask_ratio=args.mask_ratio,
    class_dropout_prob=0)

model = model.to(device)

if args.resume_path is not None:
    print(f'Resuming from {args.resume_path}')
    model.load_state_dict(torch.load(args.resume_path))
    
#
# Diffusion
#

num_timesteps = 1000
betas = get_named_beta_schedule("linear", num_timesteps)
spaced_timesteps = space_timesteps(num_timesteps=num_timesteps, section_counts=str(num_timesteps))
diffusion: SpacedDiffusion = SpacedDiffusion(
    use_timesteps=spaced_timesteps,
    betas=betas, 
    model_mean_type=ModelMeanType.EPSILON, 
    model_var_type=ModelVarType.LEARNED_RANGE, 
    loss_type=LossType.MSE)

#
# Sampling
#

def gen_samples(model, diffusion: GaussianDiffusion, num_samples = 8):
    model.eval()

    with torch.no_grad():

        x_t = torch.randn(num_samples, 4, 32, 32, device=device)
        
        if num_classes == 2:
            half = num_samples // 2
            classes_rand = torch.cat([torch.zeros(half), torch.ones(num_samples - half)], dim=0).long().to(device)
        else:
            classes_rand = torch.randint(0, num_classes, (num_samples,), device=device)

        model_kwargs = {"y": classes_rand, "enable_mask": False}

        sample = diffusion.p_sample_loop(
            model,
            x_t.shape, 
            x_t, 
            clip_denoised=False, 
            model_kwargs=model_kwargs,
            progress=True, 
            device=device)

    model.train()

    return sample

#
# Training
#

target_batch_size = 64
accumulation_steps = target_batch_size // args.batch_size

accelerator = Accelerator(
    mixed_precision='fp16', 
    gradient_accumulation_steps=accumulation_steps)

optimizer = Adan(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, max_grad_norm=1.0, fused=True)

def train_loop(model: MDTv2, optimizer, dataloader: DataLoader, diffusion: GaussianDiffusion):
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    batch_size = args.batch_size
    steps_per_epoch = args.steps_per_epoch
    batches_per_epoch = steps_per_epoch * accumulation_steps

    data_generator = dataloader_generator(dataloader)

    samples = 0

    while True:
        # Run for batches_per_epoch batches
        
        step = samples // target_batch_size

        pbar = tqdm(desc=f"Step {step}", total=args.steps_per_epoch)

        total_loss = 0
        total_mse = 0
        total_vb = 0

        for batch_idx in range(batches_per_epoch):
            # Get next batch

            with accelerator.accumulate(model):
                mean, logvar, dict = next(data_generator)
                cond = dict['y'] % num_classes

                # Sample image from latent space

                x_0 = sample(mean, logvar)

                # Get random timestep

                timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device)

                # Compute losses

                model_kwargs = {"y": cond, "enable_mask": False}
                losses_unmasked = diffusion.training_losses(model, x_0, timesteps, model_kwargs)

                model_kwargs = {"y": cond, "enable_mask": True}
                losses_masked = diffusion.training_losses(model, x_0, timesteps, model_kwargs)

                loss = losses_unmasked["loss"].mean() + losses_masked["loss"].mean()
                mse = losses_unmasked["mse"].mean() + losses_masked["mse"].mean()
                vb = losses_unmasked["vb"].mean() + losses_masked["vb"].mean()

                # Backpropagate

                accelerator.backward(loss)

                # Update weights

                optimizer.step()
                optimizer.zero_grad()

                # Update progress bar

                total_loss += loss.detach().item()
                total_mse += mse.detach().item()
                total_vb += vb.detach().item()

                samples += batch_size

            if samples % target_batch_size == 0:
                pbar.update(1)
            pbar.set_postfix({"Loss": total_loss / (batch_idx + 1),
                              "MSE": total_mse / (batch_idx + 1), 
                              "VB": total_vb / (batch_idx + 1)})
        
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
                x_0 = gen_samples(unwrapped_model, diffusion)
                decoded = decode(x_0)

                name = f"generated_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                path = os.path.join(logdir, name)
                save_sample(decoded, path)
            except Exception as e:
                print(f"Error generating sample: {e}")

train_loop(model, optimizer, dataloader, diffusion)


        
    


