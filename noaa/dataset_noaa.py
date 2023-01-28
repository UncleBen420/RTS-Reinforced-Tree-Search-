import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class NOAADataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label_filename = self.imgs[idx][:-4] + '.txt'
        label_path = os.path.join(self.root, 'labels', label_filename)

        with open(label_path) as file:
            lines = file.readlines()
        data = np.array(lines[0].split(' '), dtype=float)
        data = [int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4])]
        target = torch.as_tensor(data, dtype=torch.float32)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)