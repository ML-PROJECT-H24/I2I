import argparse
import torch
import os
import datetime
import numpy as np
from tqdm.auto import tqdm

from torchvision.utils import save_image
import torchvision

from diffusers.models import AutoencoderKL
from gaussian_diffusion import *
from mdtv2 import MDTv2

#
# Config
#

argparser = argparse.ArgumentParser()
argparser.add_argument('--model-path', type=str, default=None)
argparser.add_argument('--src-img-path', type=str, default=None)
argparser.add_argument('--strength', type=float, default=0.5)
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

model: MDTv2 = MDTv2(
    depth=12, 
    hidden_size=384, 
    patch_size=2, 
    num_heads=6, 
    num_classes=2, 
    learn_sigma=False,
    class_dropout_prob=0.0)

model = model.to(device)

model.load_state_dict(torch.load(args.model_path))

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
# Translation
#

def translate(model, x_0, noise_scheduler):
    model.eval()

    images = [x_0]

    with torch.no_grad():
        # Add noise to the image
        
        t_enc = int((args.num_inference_steps - 1) * args.strength)
        step_ratio = 1000 // args.num_inference_steps

        print(f"Encoding for {t_enc} steps, then sampling for {args.num_inference_steps - t_enc} steps")

        timesteps = np.flip(np.arange(args.num_inference_steps)[:t_enc]) * step_ratio
        timesteps = torch.tensor(timesteps, device=device)

        t_start = timesteps[0]

        print(f"timesteps: {timesteps}")

        noise = torch.randn(1, 4, 32, 32, device=device)
        cond = torch.tensor([args.cond], device=device)

        print(f"cond: {cond}")

        x_t = noise_scheduler.add_noise(x_0, noise, t_start)

        print(f"t_start: {t_start}")

        images.append(x_t)

        noise_scheduler.num_inference_steps = t_enc + 1
        noise_scheduler.timesteps = timesteps

        #noise_scheduler.set_timesteps(args.num_inference_steps)

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





        
    


