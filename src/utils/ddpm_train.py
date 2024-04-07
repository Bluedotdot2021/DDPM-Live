
import argparse
import torch
import os
import tqdm
import yaml
import numpy as np
import logging

from dataset.ddpm_dataset import DDPMDataset
from model.unet import Unet
from torch.utils.data.dataloader import DataLoader
from noise_scheduler.noise_scheduler import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    #print(config)
    ##########

    data_config = config['data_config']
    model_config = config['model_config']
    diffusion_config = config['diffusion_config']
    train_config = config['train_config']

    ddpmset = DDPMDataset(data_config['data_path'], 'train', 'png')
    ddpmloader = DataLoader(ddpmset, batch_size=train_config['batch_size'], shuffle=True)

    scheduler = NoiseScheduler(diffusion_config)

    model = Unet(model_config).to(device)
    model.train()

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    ckpt_dir = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_dir):
        print("Loading checkpoint as found one")
        model.load_state_dict(torch.load(ckpt_dir, map_location=device))

    logging.basicConfig(filename=os.path.join(train_config['task_name'], "log.txt"), filemode='w', format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    optimizer = torch.optim.Adam(model.parameters(), lr = train_config['lr'])
    criterion = torch.nn.MSELoss()

    logging.basicConfig(filename=os.path.join(train_config['task_name'], train_config['log_name']))

    for epoch_idx in range(train_config['num_epochs']):
        losses = []
        for (imgs, labels) in tqdm.tqdm(ddpmloader):
            optimizer.zero_grad()

            imgs = imgs.float().to(device) # imgs: [b, c, h, w]
            noise = torch.randn_like(imgs).to(device) # noise: [b, c, h, w]
            t = torch.randint(0, diffusion_config['num_timesteps'], (imgs.shape[0],)).to(device) # t: [b]

            noisy_imgs = scheduler.add_noise(imgs, noise, t)
            #print("imgs.device:{}, noisy_imgs.device:{}, t.device:{}".format(imgs.device, noisy_imgs.device, t.device))
            #print("model.device:{}".format(next(model.parameters()).device))
            noise_pred = model(noisy_imgs, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Epoch: {} finished | Loss: {}".format(epoch_idx, np.mean(losses)))
        logger.info("Epoch:{}, Loss:{}".format(epoch_idx, np.mean(losses)))
        torch.save(model.state_dict(), ckpt_dir)

    print("Done training...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument("--config_path", default="../config/default.yaml")
    args = parser.parse_args()

    print("parse config_path:{}".format(args.config_path))
    train(args)