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
argparser.add_argument('--num-samples', type=int, default=8)
argparser.add_argument('--sample-steps', type=int, default=250)
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--cond', type=int, default=0)
argparser.add_argument('--logdir', type=str, default='logs')

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logdir = os.path.join(args.logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(logdir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True


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
    hidden_size=768, 
    patch_size=2, 
    num_heads=12,
    num_classes=2, 
    learn_sigma=True,
    class_dropout_prob=0.0)

model = model.to(device)

model.load_state_dict(torch.load(args.model_path))

#
# Diffusion
#

num_timesteps = 1000
betas = get_named_beta_schedule("linear", num_timesteps)
spaced_timesteps = space_timesteps(num_timesteps=num_timesteps, section_counts=str(args.sample_steps))
diffusion: SpacedDiffusion = SpacedDiffusion(
    use_timesteps=spaced_timesteps,
    betas=betas, 
    model_mean_type=ModelMeanType.EPSILON, 
    model_var_type=ModelVarType.LEARNED_RANGE, 
    loss_type=LossType.MSE)

#
# Translation
#

def gen_samples(model, diffusion: SpacedDiffusion):
    model.eval()

    cond = torch.tensor([args.cond], device=device)
    
    with torch.no_grad():
        # Add noise to the image

        x_t = torch.randn((args.num_samples, 4, 32, 32), device=device)

        # Diffuse the image

        model_kwargs = {"y": cond, "enable_mask": False}

        samples = diffusion.p_sample_loop(
            model, x_t.shape, x_t, clip_denoised=False, model_kwargs=model_kwargs, progress=True)

    model.train()

    # Add all images into a single tensor
    return samples


images = gen_samples(model, diffusion)
decoded = decode(images)

for i in range(args.num_samples):
    img = decoded[i]
    save_sample(img, os.path.join(logdir, f'sample_{i}.png'))





        
    


