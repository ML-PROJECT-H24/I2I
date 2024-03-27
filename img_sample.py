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
argparser.add_argument('--model-path', type=str, default=None)
argparser.add_argument('--num-samples', type=int, default=4)
argparser.add_argument('--num-iterations', type=int, default=100)
argparser.add_argument('--logdir', type=str, default='logs')

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logdir = os.path.join(args.logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(logdir, exist_ok=True)

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

num_classes = 1000
model: MDTv2 = MDTv2(depth=12, hidden_size=384, patch_size=2, num_heads=6, num_classes=num_classes, learn_sigma=False, mask_ratio=args.mask_ratio)
#model: MDTv2 = MDTv2(depth=12, hidden_size=768, patch_size=2, num_heads=12, num_classes=num_classes, learn_sigma=False, mask_ratio=args.mask_ratio)
model = model.to(device)

print(f'Model from {args.resume_path}')
model.load_state_dict(torch.load(args.model_path))

#
# Diffusion
#

num_train_timesteps = 1000
noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

#
# Sampling
#

def gen_samples(model, noise_scheduler, num_samples = 8, cgf=False):
    model.eval()

    with torch.no_grad():

        x_t = torch.randn(num_samples, 4, 32, 32, device=device)
        classes_rand = torch.randint(0, num_classes, (num_samples,), device=device)

        if cgf: # Classifier free guidance
            classes_null = torch.tensor([num_classes] * num_samples, device=device)
            classes_all = torch.cat([classes_rand, classes_null], 0)
            x_t = torch.cat([x_t, x_t], 0)
        else: # Random classes
            classes_all = classes_rand

        noise_scheduler.set_timesteps(args.num_iterations)

        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            # predict the noise residual
            with torch.no_grad():
                # Classifier free guidance
                tt = torch.tensor([t], device=device)
                noise_pred = model.forward_with_cfg(x_t, tt, classes_all)

            # compute the previous noisy sample x_t -> x_t-1
            x_t = noise_scheduler.step(noise_pred, t, x_t).prev_sample
        
        if cgf:
            x_0, _ = torch.chunk(x_t, 2, 0)
        else:
            x_0 = x_t

    model.train()

    return x_0

samples = gen_samples(model, noise_scheduler, num_samples=args.num_samples, cgf=False)

decoded = decode(samples)

for i in range(args.num_samples):
    save_sample(decoded[i], os.path.join(logdir, f'sample_{i}.png'))


        
    


