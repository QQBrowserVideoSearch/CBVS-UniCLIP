import os
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import torch
import lmdb
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
import torchvision.datasets as datasets
from uniclip.clip import tokenize
import pickle
from tqdm import tqdm
import requests
from uniclip.clip import image_transform

def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class EvalDataset(Dataset):
    def __init__(self, cbvs_path):
        self.text_ids = []
        self.querys = []
        self.imgs = []
        self.preprocess = image_transform(224)

        with open(cbvs_path, 'r') as fin:
            for line in tqdm(fin):
                li = line.strip().split('\t')
                query = li[0]
                docid = li[1]
                
                img_path = li[2]
                label = li[3]

                self.text_ids.append(f'{label}\t{query}\t{docid}')
                self.querys.append(query)
                self.imgs.append(img_path)

    def __len__(self):
        return len(self.querys)

    def __getitem__(self, idx):
        text_id = self.text_ids[idx]
        text = self.querys[idx]
        text = tokenize(text, context_length=12)[0]
        image_path = self.imgs[idx]

        try:
            # from online
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))

            # from local
            # docid = text_id.split('\t')[-1]
            # image = Image.open(f'/search/odin/xsqiao/UniOCR/Code/pnr_utils/imgs/{docid}.jpg')

            image = self.preprocess(image)
        except Exception:
            image = torch.zeros((3, 224, 224))

        return text_id, image, text


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


def get_eval_dataset(cbvs_path, batch_size):
    dataset = EvalDataset(cbvs_path)
    num_samples = len(dataset)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)