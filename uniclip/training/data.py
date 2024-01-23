from LAC import LAC

from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass
import pickle
import random
import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from tqdm import tqdm
from uniclip.clip import _tokenizer
from uniclip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class CBVSDataset(Dataset):
    def __init__(self, cbvs_path, split="train", max_txt_length=64, use_augment=False, resolution=224):
        super(CBVSDataset, self).__init__()

        self.split = split
        self.max_txt_length = max_txt_length        
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)
        
        self.sim_database = {}
        self.ocr_database = []

        self.lac = LAC(mode='rank')
        
        # 数据集
        self.titles = []
        self.urls = []
        self.ocrs = []

        print('Loading CBVS dataset')
        with open(cbvs_path, 'r') as fin:
            for line in tqdm(fin):
                li = line.strip().split('\t')
                if len(li) < 3:
                    continue
                title = li[0]
                url = li[1]
                ocr = li[2]
                self.titles.append(title)
                self.urls.append(url)
                self.ocrs.append(ocr)
        
        # OCR负样本
        print('Loading OCR samples')
        with open('datasets/cbvs10m-HNSW', 'r') as fin:
            for line in tqdm(fin):
                li = line.strip().split('\t')
                self.sim_database[li[0]] = li[1:]
                self.ocr_database.append(li[0])
        
        self.dataset_len = len(self.titles)


    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                             input_size=resolution,
                             scale=(0.9, 1.0),
                             is_training=True,
                             color_jitter=None,
                             auto_augment='original',
                             interpolation='bicubic',
                             mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711),
                         )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform


    def __len__(self):
        return len(self.titles)


    def get_core_terms(self, text):
        """
        :param lac: 百度开源工具包 lac = LAC(mode='rank')
        :param text: 输入的文本
        :return: 核心词list
        """
        core_terms = []
        text_lis = [u'' + text]
        rank_result = self.lac.run(text_lis)

        if rank_result and len(rank_result[0]) == 3:
            words = rank_result[0][0]  # 分词list
            term_weights = rank_result[0][2]  # 词重要度list

            for index in range(len(words)):
                if term_weights[index] > 2:
                    core_terms.append(words[index])

        return core_terms

    def __getitem__(self, index):
        try:
            response = requests.get(self.urls[index])
            image = Image.open(BytesIO(response.content))
            image = self.transform(image)
        except:
            image = torch.zeros((3, 224, 224))
        
        query = self.titles[index]
        ocr = self.ocrs[index]

        try:
            new_query = self.get_core_terms(query)
            new_query = " ".join(new_query).strip()
            if len(new_query) > 2:
                query = new_query
        except Exception:
            pass
        
        ocr_presences = 1 if ocr != '\\N' else 0
        query = tokenize([_preprocess_text(query)], context_length=self.max_txt_length)[0]

        # 随机在最接近的10个里面选择1个作为负样本
        if ocr == '\\N':
            fake_ocr = random.choice(self.ocr_database)
            ocr_semantic = 0
        else:
            if random.random() < 0.7:
                fake_ocr = ocr
                ocr_semantic = 1
            else:
                if ocr in self.sim_database:
                    fake_ocr = random.choice(self.sim_database[ocr])
                    ocr_semantic = 1 if fake_ocr==ocr else 0
                else:
                    fake_ocr = ocr
                    ocr_semantic = 1

        ocr =  tokenize([_preprocess_text(ocr)], context_length=20)[0]
        fake_ocr =  tokenize([_preprocess_text(fake_ocr)], context_length=20)[0]
        eos_index = query.numpy().tolist().index(_tokenizer.vocab['[SEP]'])

        return image, query, ocr, ocr_presences, fake_ocr, ocr_semantic, eos_index


def pad_dataset(dataset, global_batch_size):
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: CBVSDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = CBVSDataset(
        db_path, 
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    ) 

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args, 
            is_train=True,  
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    if args.val_data:
        data["val"] = get_dataset(
            args, 
            is_train=False, 
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    return data
