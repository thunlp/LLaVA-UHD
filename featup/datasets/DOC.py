from os.path import join

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset

class Doc(Dataset):
    def __init__(self,
                 root,
                 split,
                 transform,
                 target_transform,
                 subset=None):
        super(Doc, self).__init__()
        self.split = split
        self.root = join(root, "Doc")
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset

        if self.subset is None:
            self.image_list = "Doc20000.txt"
        elif self.subset == 'Doc_validation50':
            self.image_list = "Doc_validation50.txt"

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train"],
            "val": ["val"],
            "train+val": ["train", "val"]
        }

        self.image_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", self.image_list), "r") as f:
                img_names = [fn.rstrip() for fn in f.readlines()]
                for img_name in img_names:
                    self.image_files.append(join(self.root, "images", img_name))
                    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        batch = {}
        img = self.transform(Image.open(image_path).convert("RGB"))
        batch["img"] = img
        batch["img_path"] = image_path
        return batch
