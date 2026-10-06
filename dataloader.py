import pathlib as pl
from PIL import Image
from io import BytesIO
import glob

import yaml
import ast
import json
import pandas as pd
import torch
from datasets import load_dataset
from torchvision.io import decode_image, ImageReadMode
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

def pil_read(image_path: str):
    with open(image_path, "rb") as f:
        image = Image.open(f)
        image = image.convert("RGB")
    return image

class ImageNet(Dataset):
    def __init__(self, data_root: pl.Path, data_split: str, mapping_file: str, dtype=torch.bfloat16, class_mapping:dict=None):

        if class_mapping is not None:
            self.class_mapping = class_mapping
        else:
            with open(mapping_file, "r") as reader:
                mapping_str = reader.read()
            mapping_dict = ast.literal_eval(mapping_str)
            self.class_mapping = {}
            for i, (k, v) in enumerate(mapping_dict.items()):
                self.class_mapping[k] = i
        # this is a place holder for the imagenet dataset, which is not used in the current experiments.

        data_folder = pl.Path(data_root[data_split])
        self.data = list(data_folder.glob("*.JPEG"))
        if data_split == "train":
            self.transforms = v2.Compose([
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(dtype, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = v2.Compose([
                v2.Resize(size=(256), antialias=True),
                v2.CenterCrop(size=(224, 224)),
                v2.ToDtype(dtype, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.data)

    def get_class_mapping(self):
        return self.class_mapping
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        word_id = img_path.name.split("_")[0]
        image = decode_image(img_path, mode=ImageReadMode.RGB)
        image = self.transforms(image)
        label_id = self.class_mapping.get(word_id, -1)  # the folder name is like "n01440764", we need to convert it to the corresponding label id.
        return image, label_id

class CIFAR10(Dataset):
    def __init__(self, data_root: pl.Path, data_split: str, transforms=None, class_mapping=None):
        data_root = pl.Path(data_root)
        sub_folder = data_root / "plain_text"
        if data_split == "train":
            par_file = sub_folder / "train-00000-of-00001.parquet"
        else:
            par_file = sub_folder / "test-00000-of-00001.parquet"
        
        self.data = pd.read_parquet(par_file)
        self.transforms = transforms
        self.class_mapping = class_mapping # this is not used for the cifar dataset, but a place holder.

    def get_class_mapping(self):
        return self.class_mapping 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_bytes, label = row['img']['bytes'], row['label']
        pil_image = Image.open(BytesIO(image_bytes))
        #image_tensor_bytes = torch.frombuffer(image_bytes, dtype=torch.uint8)
        image = v2.functional.pil_to_tensor(pil_image)

        if self.transforms:
            image = self.transforms(image)
        return image, label

class CIFAR100(Dataset):
    def __init__(self, data_root: pl.Path, data_split: str, dtype=None, class_mapping=None):
        data_root = pl.Path(data_root)
        if data_split == "train":
            par_file = data_root / "train-00000-of-00001.parquet"
            self.transforms = v2.Compose([
                v2.RandomHorizontalFlip(0.5),
                v2.RandomRotation(20),
                v2.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
                v2.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1),
                v2.ToDtype(dtype, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                v2.RandomErasing(p=0.75,scale=(0.02, 0.1),value = 1.0, inplace = False)
            ]) 
        else:
            par_file = data_root / "test-00000-of-00001.parquet"
            self.transforms = v2.Compose([
                v2.ToDtype(dtype, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]) 
        
        self.data = pd.read_parquet(par_file)
        self.class_mapping = class_mapping # this is not used for the cifar dataset, but a place holder.

    def get_class_mapping(self):
        return self.class_mapping 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_bytes, label = row['img']['bytes'], row['fine_label']
        pil_image = Image.open(BytesIO(image_bytes))
        #image_tensor_bytes = torch.frombuffer(image_bytes, dtype=torch.uint8)
        image = v2.functional.pil_to_tensor(pil_image)

        image = self.transforms(image)
        return image, label

class TinyImageNet(Dataset):
    def __init__(self, data_root: pl.Path, data_split: str, transforms=None, class_mapping:dict=None):
        self.transforms = transforms
        data_root = pl.Path(data_root)
        sub_folder = data_root / data_split
        class_set = set()
        self.data = []

        if data_split == "train":
            for class_folder in sub_folder.glob("*"):
                class_name = class_folder.name
                class_set.add(class_name)
                for f in class_folder.glob("images/*"):
                    self.data.append((f, class_name))

            self.wid_to_label = {}
            self.class_mapping = {}

            with open(data_root/"words.txt", "r") as reader:
                for line in reader:
                    wid, name = line.rstrip("\n").split("\t")
                    if wid in class_set:
                        # the index for each class is fixed given the order in the text file.
                        self.class_mapping[wid] = len(self.wid_to_label) 
                        self.wid_to_label[wid] = name
        
        elif data_split == "val":
            assert class_mapping, "The WordNet ID to index is required,\
                which should be created in the training data loader."
            self.class_mapping = class_mapping
            img_to_wid = {}
            with open(sub_folder/"val_annotations.txt", "r") as reader:
                for line in reader:
                    img_name, wid = line.rstrip("\n").split("\t")[:2]
                    img_to_wid[img_name] = wid

            for f in sub_folder.glob("images/*"):
                wid = img_to_wid[f.name]
                self.data.append((f, wid))
        
    def get_class_mapping(self):        
        return self.class_mapping

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, wid = self.data[idx]
        image = pil_read(img_path)
        if self.transforms:
            image = self.transforms(image)
        label_id = self.class_mapping.get(wid, -1)
        return image, label_id

class ImageWoof(Dataset):
    def __init__(self, data_root: pl.Path, data_split: str, transforms=None, class_mapping:dict=None):
        # this is a place holder for the imagewoof dataset, which is not used in the current experiments.
        self.data_folder = pl.Path(data_root) / data_split

        if class_mapping is not None:
            self.class_mapping = class_mapping
        else:
            self.class_mapping = {}

        self.data = []

        for i, folder in enumerate(self.data_folder.glob("*")):
            class_name = folder.name
            if class_mapping is None:
                if class_name not in self.class_mapping:
                    self.class_mapping[class_name] = len(self.class_mapping)

            label = self.class_mapping[class_name]
            for f in folder.glob("*.JPEG"):
                self.data.append((f, label))

        self.transforms = transforms
    
    def get_class_mapping(self):
        return self.class_mapping

    def __len__(self):
        return len(self.data)   
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = decode_image(img_path, mode=ImageReadMode.RGB)
        if self.transforms:
            image = self.transforms(image)
        return image, label

def build_dataloader(data_name: str, 
    data_conf,
    data_split:str, 
    image_size:int=32,
    batch_size:int=64, 
    num_workers:int=4, 
    dtype:torch.dtype=torch.bfloat16, 
    class_mapping=None,
    drop_last:bool=False,
    ):

    if data_split == "train":
        transforms = v2.Compose([
            #v2.Resize(size=(image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            # RandAugment with 2 layers and standard magnitude 9
            v2.RandAugment(num_ops=2, magnitude=9, interpolation=v2.InterpolationMode.BICUBIC),
            # Required for mixing and tensor transformations
            v2.ToImage(),
            v2.ToDtype(dtype, scale=True),
            # Standard ImageNet normalization values
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Random Erasing operates on float tensors
            v2.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
        ])
        shuffle = True
    else:
        transforms = v2.Compose([
            v2.Resize(size=(image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            #v2.CenterCrop(size=(image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(dtype, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        shuffle = False 

    if data_name == "tiny-imagenet":    
        dataset = TinyImageNet(data_conf["data_path"], data_split, transforms, class_mapping)
    elif data_name == "cifar10":
        dataset = CIFAR10(data_conf["data_path"], data_split, dtype, class_mapping)
    elif data_name == "cifar100":
        dataset = CIFAR100(data_conf["data_path"], data_split, dtype, class_mapping)
    elif data_name == "imagewoof":
        dataset = ImageWoof(data_conf["data_path"], data_split, transforms, class_mapping)
    elif data_name == "imagenet":
        dataset = ImageNet(data_conf["data_path"], data_split, data_conf["mapping_file"], dtype, class_mapping)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return dataloader

def process_hf_dataset(examples, transforms):
    examples["pixel_values"] = [transforms(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

def build_tiny_imagenet(data_path: str, data_split:str, batch_size:int=64, num_workers:int=4, dtype:torch.dtype=torch.bfloat16):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data = load_dataset("arrow", data_files=data_path)[data_split]
    data.set_transform(lambda examples: process_hf_dataset(examples, transforms))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=(data_split=="train"), num_workers=num_workers)
    return dataloader

def build_imagenet(data_path: str, data_split:str, batch_size:int=64, num_workers:int=4, dtype:torch.dtype=torch.bfloat16):
    # this is a place holder for the imagenet dataset, which is not used in the current experiments.
    transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(dtype, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) if data_split == "train" else v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(256), antialias=True),
        v2.CenterCrop(size=(224, 224)),
        v2.ToDtype(dtype, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_imagenet_hf(data_path: str, data_split:str, batch_size:int=64, num_workers:int=4, dtype:torch.dtype=torch.bfloat16):
    # this is a place holder for the imagenet dataset, which is not used in the current experiments.
    transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(dtype, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) if data_split == "train" else v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(256), antialias=True),
        v2.CenterCrop(size=(224, 224)),
        v2.ToDtype(dtype, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if data_split == "train":
        files = glob.glob(f"{data_path}/train-*.parquet")
    else:
        files = glob.glob(f"{data_path}/validation-*.parquet")

    data = load_dataset("parquet", data_files=files)['train']
    data.set_transform(lambda examples: process_hf_dataset(examples, transforms))
    #data.map(lambda examples: process_hf_dataset(examples, transforms))
    return data

cnt = 0
def save_image(examples):
    global cnt
    if examples["label"] == -1:
        cnt += 1
    #for i in examples["label"]:
        #v2.functional.to_pil_image(image).save(f"test_{cnt}.jpg")
    #    cnt += 1

if __name__ == "__main__":
    magic_test = "/home/sen/workspace/dataset/CLASSIFICATION/image_net/data"
    dataset = build_imagenet(magic_test, "train", batch_size=64, num_workers=4, dtype=torch.bfloat16)
    dataset.map(save_image)
    print (cnt)
