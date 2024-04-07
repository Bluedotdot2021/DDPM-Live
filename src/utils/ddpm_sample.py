
import argparse
import yaml
import os
import torch
import tqdm
import torchvision

from model.unet import Unet
from noise_scheduler.noise_scheduler import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, model_config, diffusion_config, train_config ):
    xt = torch.randn((train_config['num_samples'], model_config['img_channels'], model_config['img_size'], model_config['img_size'])).to(device)

    sample_dir = "ddpm_sample"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    for i in tqdm.tqdm(reversed(range(diffusion_config['num_timesteps']))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        xt, x0 = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        imgs = torch.clamp(xt, -1.,1.).detach().cpu()
        imgs = (imgs + 1)/2
        grid_xt = torchvision.utils.make_grid(imgs, nrow=10)
        grid_img = torchvision.transforms.ToPILImage()(grid_xt)
        grid_img.save(os.path.join(sample_dir,"sample_{}.png".format(i)))
        grid_img.close()

    print("Done sampling...")


def infer(args):
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    print(config)
    ####################

    model_config = config['model_config']
    diffusion_config = config['diffusion_config']
    train_config = config['train_config']

    ckpt_dir = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    assert os.path.exists(ckpt_dir), print("No checkpoint file found")

    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(ckpt_dir, map_location=device))
    model.eval()

    scheduler = NoiseScheduler(diffusion_config)

    with torch.no_grad():
        sample(model, scheduler, model_config, diffusion_config, train_config )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm sampling...")
    parser.add_argument("--config_path", default="../config/default.yaml")
    args = parser.parse_args()

    print("parse config_path:{}".format(args.config_path))
    infer(args)