import argparse

import torch
import PIL 
import datetime
import torchvision

import diffusion2
import latent_dataset
import AttnUnet

from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

#
# Config
#

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='data/LATENT_DATASET/LATENT_DATASET')
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--batch-size', type=int, default=1)
argparser.add_argument('--resume-path', type=str, default=None)
argparser.add_argument('--gen-only', action='store_true', default=False)

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter()

#
# Dataset
#

dataset = latent_dataset.LatentImageDataset(args.data_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

#
# Model
#

# Simple MLP model
# input = (4, 32, 32)
# hidden = [256, 256]
# output = (4, 32, 32)

# Simple MLP model
model = AttnUnet.Unet(dim=32, channels=4, dim_mults=(1, 2, 4, 8, 8), learned_variance=True).to(device)


if args.resume_path is not None:
    print(f'Resuming from {args.resume_path}')
    model.load_state_dict(torch.load(args.resume_path))

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
    #x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return x

#
# Diffusion
#

timesteps = 1000
betas = diffusion.get_linear_beta_schedule(timesteps)
diff = diffusion.GaussianDiffusion(betas)

#
# Training
#

if not args.gen_only:

  print('Training...')

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  def save_model():
      torch.save(model.state_dict(), 'model.pth')

  best = None

  accumulated_loss = 0

  for step, (mean, logvar) in enumerate(dataloader):

      optimizer.zero_grad()

      mean, logvar = mean.to(device), logvar.to(device)
      
      t = torch.randint(0, timesteps, (1,)).to(device)
      noise = torch.randn(1, 4, 32, 32).to(device)

      x_0 = sample(mean, logvar)

      losses = diff.training_losses(model, x_0, t)

      loss = losses['loss'].mean()

      loss.backward()
      optimizer.step()

      accumulated_loss += loss.item()

      if step % 100 == 0:
          print(f'Step: {step}, Loss: {accumulated_loss / 100}')
          accumulated_loss = 0

      if step % 1000 == 0:
          save_model()

  save_model()

#
# Generation
#
  
cfg_scale = 4.0
pow_scale = 0.01 # large pow_scale increase the diversity, small pow_scale increase the quality.

print('Generating...')

model = model.eval()
# Create sampling noise
z = torch.randn(1, 4, 32, 32, device=device)
z = torch.cat([z, z], dim=0)
# Generate images
x = diff.p_sample_loop(model, z.shape, z, clip_denoised=False, progress=True, device=device)
print(x.shape)
x, _ = x.chunk(2, dim=0) # Remove the variance 
print(x.shape)

decoded = decode(x)
# Save image
name = f"generated_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
save_image(decoded, name, nrow=3, normalize=True, value_range=(-1, 1))



    




    






