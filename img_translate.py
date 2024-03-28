import argparse
import torch
import PIL 
import os
import datetime
import numpy as np
from tqdm.auto import tqdm

import latent_dataset
from torchvision.utils import save_image
import torchvision
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
argparser.add_argument('--src-img-path', type=str, default=None)
argparser.add_argument('--start-t', type=int, default=200)
argparser.add_argument('--num-inference-steps', type=int, default=50)
argparser.add_argument('--cond', type=int, default=0)
argparser.add_argument('--logdir', type=str, default='logs')

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logdir = os.path.join(args.logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(logdir, exist_ok=True)

#
# Load Image
#

def center_crop_square(image: torch.Tensor):
    # Crop center of image to be square using pytorch
    _, width, height = image.shape
    new_size = min(width, height)

    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2

    return image[:, left:right, top:bottom]

def resize_square(image: torch.Tensor, size: int): 
  return torchvision.transforms.functional.resize(image, size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

image_file = args.src_img_path
TARGET_SIZE = 256

image = torchvision.io.read_image(image_file, mode=torchvision.io.image.ImageReadMode.RGB)
image = image.to(device=device, dtype=torch.float32)
image = center_crop_square(image)
image = resize_square(image, TARGET_SIZE)
image = (image / 127.5) - 1
image = image.unsqueeze(0)  

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
model: MDTv2 = MDTv2(depth=12, hidden_size=384, patch_size=2, num_heads=6, num_classes=num_classes, learn_sigma=False)
model = model.to(device)

model.load_state_dict(torch.load(args.model_path))

#
# Diffusion
#

num_train_timesteps = 1000
noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

#
# Translation
#

def translate(model, x_0, noise_scheduler):
    model.eval()

    images = [x_0]

    # Add noise to the image

    start_t = args.start_t - 1

    noise = torch.randn(1, 4, 32, 32, device=device)
    timesteps = torch.tensor([start_t], device=device)
    cond = torch.tensor([args.cond], device=device)
    
    x_t = noise_scheduler.add_noise(x_0, noise, timesteps)

    images.append(x_t)

    # Conditionally sample the noise

    with torch.no_grad():
        noise_scheduler.num_inference_steps = args.num_inference_steps

        step_ratio = start_t // args.num_inference_steps

        timesteps = (np.arange(0, args.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        noise_scheduler.timesteps = torch.from_numpy(timesteps).to(device)

        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            with torch.no_grad():
                tt = torch.tensor([t], device=device)
                noise_pred = model(x_t, tt, cond, enable_mask=False)

            # compute the previous noisy sample x_t -> x_t-1
            x_t = noise_scheduler.step(noise_pred, t, x_t).prev_sample

            images.append(x_t)

    model.train()

    # Add all images into a single tensor
    return images


mean, logvar = encode(image)
image = sample(mean, logvar)
images = translate(model, image, noise_scheduler)

for i, image in enumerate(images):
    decoded = decode(image)
    save_sample(decoded, os.path.join(logdir, f'sample_{i}.png'))





        
    


