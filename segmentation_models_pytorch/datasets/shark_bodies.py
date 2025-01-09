import os
import torch
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO 

class SharkBodies(torch.utils.data.Dataset):

    def __init__(self, root, mode="train"):
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode

        # path to images/masks stored at root
        self.images_directory = os.path.join(self.root, "images") # (?)
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps") # (?)

        # filenames according to the chosen split
        self.filenames = self._read_split()

    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        # define filenames, image paths, and annotation mask paths
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png") #(?)

        # load the image and mask
        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        # store the image, mask, and trimap in a dictionary
        sample = dict(image=image, mask=mask, trimap=trimap)

        return sample

    def _read_split(self): # checked 01/08 good
        if self.mode == "train":
            split_filename = "train.json"
        elif self.mode == "valid":
            split_filename = "val.json"
        else:
            split_filename = "test.json"

        split_filepath = os.path.join(self.root, "annotations", split_filename)

        with open(split_filepath, 'r') as f:
            split_data = json.load(f)

        filenames = [item["image_id"] for item in split_data]

        return filenames


class SimpleSharkBodies(SharkBodies):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)

        # resize images, masks, and trimaps to 256x256
        image = np.array(
            Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
        )
        trimap = np.array(
            Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST)
        )

        # convert from HWC to CWH
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)  # Add channel dimension for mask
        sample["trimap"] = np.expand_dims(trimap, 0)  # Add channel dimension for trimap

        return sample
