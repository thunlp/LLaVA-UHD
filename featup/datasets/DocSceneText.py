from os.path import join

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset

class DocSceneText(Dataset):
    def __init__(self,
                 root,
                 split,
                 transform,
                 target_transform,
                 subset=None):
        super(DocSceneText, self).__init__()
        self.split = split
        self.root = join(root, "224DocSceneText")
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset

        if self.subset is None:
            self.image_list = "224docSceneText.txt"

        self.image_files = []
        with open(join(self.root, "curated", self.image_list), "r") as f:
            img_names = [fn.rstrip() for fn in f.readlines()]
            for img_name in img_names:
                self.image_files.append(join(self.root, img_name))      
                    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        batch = {}
        img = self.transform(Image.open(image_path).convert("RGB"))
        batch["img"] = img
        batch["img_path"] = image_path
        return batch
