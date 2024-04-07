
import torch
import os
import glob
import PIL
import torchvision

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class DDPMDataset(Dataset):
    def __init__(self, data_path, split, img_ext):
        super().__init__()
        self.split = split
        self.img_ext = img_ext

        self.imgs, self.labels = self._load_imgs(data_path, split, img_ext)

    def _load_imgs(self, data_path, split, img_ext): # data_path:"data", split:"train", img_ext:"png"
        data_dir = os.path.join(data_path, split) # "data" + "train" -> "data/train" ("data\\train")
        assert os.path.exists(data_dir), print("Data directory:{} not found".format(data_dir))

        imgs = []
        labels = []
        for d_name in os.listdir(data_dir):
            for f_name in glob.glob(os.path.join(data_dir, d_name, "*.{}".format(img_ext))):
                imgs.append(f_name)
                labels.append((int(d_name)))
        print("Found {} images for {}".format(len(imgs), split))
        return imgs, labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = PIL.Image.open(self.imgs[index])
        img = torchvision.transforms.ToTensor()(img)

        img = img*2 - 1
        return img, self.labels[index]

'''
ddpmset = DDPMDataset("data", "train", "png")
ddpmloader = DataLoader(ddpmset, batch_size=36, shuffle=True)

for batch_idx, (imgs, labels) in enumerate(ddpmloader):
    if batch_idx > 10:
        break

    imgs = (imgs + 1)  / 2
    grid = torchvision.utils.make_grid(imgs, nrow=6)
    grid_imgs = torchvision.transforms.ToPILImage()(grid)
    grid_imgs.save(os.path.join("merge_{}.png".format(batch_idx)))
    grid_imgs.close()
'''