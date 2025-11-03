import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import xml.etree.ElementTree as ET
import requests


def get_imagenet_transform():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class ImageNetValDataset(Dataset):
    imagenet_classes = None

    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.annotation_filenames = sorted(os.listdir(annotation_dir))

        # Create class mapping (Synset ID → class index)
        self.synset_to_label = self.get_imagenet_label_map()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, self.annotation_filenames[idx])

        image = Image.open(img_path).convert("RGB")

        synset_id = self.parse_annotation(ann_path)
        label = self.synset_to_label.get(synset_id, -1)

        if self.transform:
            image = self.transform(image)

        return image, label

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        synset_id = root.find("object").find("name").text
        return synset_id

    @classmethod
    def load_imagenet_classes(cls):
        if cls.imagenet_classes is None:
            url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            response = requests.get(url)
            cls.imagenet_classes = response.json()

    def get_imagenet_label_map(self):
        self.load_imagenet_classes()
        return {v[0]: int(k) for k, v in self.imagenet_classes.items()}

    @classmethod
    def to_class_names(cls, label_idx):
        cls.load_imagenet_classes()
        return cls.imagenet_classes[str(label_idx)][1]
    
    def get_indices_by_synset(self, target_synset):
        indices = []
        for idx, ann_file in enumerate(self.annotation_filenames):
            ann_path = os.path.join(self.annotation_dir, ann_file)
            synset_id = self.parse_annotation(ann_path)
            if synset_id == target_synset:
                indices.append(idx)
        return indices
