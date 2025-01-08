import os
import torch
import numpy as np
from PIL import Image


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        """
        Args:
            root (str): Path to the root directory containing the dataset.
            mode (str): Mode to load data (train, valid, or test). Default is "train".
            transform (callable, optional): A function/transform to apply to the sample.
        """
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        # Paths to images and masks
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        # Read filenames according to the chosen split (train/valid/test)
        self.filenames = self._read_split()

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieves a sample (image, mask, trimap) at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the image, mask, and trimap.
        """
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # Load the image and mask
        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        # Store the image, mask, and trimap in a dictionary
        sample = dict(image=image, mask=mask, trimap=trimap)

        # Apply transformations if provided
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        """
        Preprocesses the mask to convert it to the required format.
        Since the values are already 0 (background) and 1 (foreground), 
        no remapping is necessary.

        Args:
            mask (numpy.ndarray): The mask image.

        Returns:
            numpy.ndarray: The preprocessed mask.
        """
        return mask.astype(np.float32)

    def _read_split(self):
        """
        Reads the split file (train, validation, or test) and returns the filenames.

        Returns:
            list: A list of filenames corresponding to the selected split.
        """
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)

        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")

        filenames = [x.split(" ")[0] for x in split_data]

        # Split filenames into training and validation sets
        if self.mode == "train":  # 90% for training
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]

        return filenames


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        """
        Retrieves a sample and applies additional transformations like resizing.
        """
        sample = super().__getitem__(*args, **kwargs)

        # Resize images, masks, and trimaps to 256x256
        image = np.array(
            Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
        )
        trimap = np.array(
            Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST)
        )

        # Convert image from HWC (height, width, channels) to CHW (channels, height, width)
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)  # Add channel dimension for mask
        sample["trimap"] = np.expand_dims(trimap, 0)  # Add channel dimension for trimap

        return sample
