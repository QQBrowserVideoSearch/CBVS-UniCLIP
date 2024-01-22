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

import lmdb
import pickle
import random
import numpy as np


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
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        self.lmdb_path = lmdb_path

        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(lmdb_pairs, split)
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(lmdb_imgs, split)

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))
        logging.info("{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples))

        super(LMDBDataset, self).__init__()

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        
        self.global_batch_size = 1 # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split
        self.max_txt_length = max_txt_length        

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

        self.ocr_database = []     
        self.sim_database = {}

        self.lac = LAC(mode='rank')
        
        # OCR负样本
        with open('/search/odin/1_ceceliali/8_mmfuse_model/1_data_mining/1_hnswlib_mining/1204_file_merge_query_distance', 'r') as fin: # 1000W
            for line in fin:
                li = line.strip().split('\t')
                raw_ocr = li[0]
                hard_ocr = li[1]
                if raw_ocr not in self.sim_database:
                    self.sim_database[raw_ocr] = [hard_ocr]
                    self.ocr_database.append(raw_ocr)
                else:
                    if len(self.sim_database[raw_ocr]) < 10:
                        self.sim_database[raw_ocr].append(hard_ocr)


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

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

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
            # pos_tags = rank_result[0][1]  # 词性list
            term_weights = rank_result[0][2]  # 词重要度list: 3表示核心词

            for index in range(len(words)):
                if term_weights[index] > 2:
                    core_terms.append(words[index])

        return core_terms

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode('utf-8')).tobytes())
        image_id, text_id, raw_text = pair

        try:
            image_b64 = self.txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
            image_b64 = image_b64.decode(encoding="utf8", errors="ignore")

            image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized
            image = self.transform(image)
        except:
            print('error img idx: ', image_id)
            image = torch.zeros((3, 224, 224))
        
        query, ocr = raw_text.split('xiaozhequ')


        try:
            new_query = self.get_core_terms(query)
            new_query = " ".join(new_query).strip()
            if len(new_query) > 2:
                query = new_query
        except Exception:
            pass
        
        ocr_exists = 1 if ocr != 'Null' else 0
        query = tokenize([_preprocess_text(query)], context_length=self.max_txt_length)[0]


        # 随机在最接近的10个里面选择1个作为负样本
        if ocr == 'Null':
            fake_ocr = random.choice(self.ocr_database)
            ocr_content = 0
        else:
            if random.random() < 0.7:
                fake_ocr = ocr
                ocr_content = 1
            else:
                if ocr in self.sim_database:
                    fake_ocr = random.choice(self.sim_database[ocr])
                    ocr_content = 1 if fake_ocr==ocr else 0
                else:
                    fake_ocr = ocr
                    ocr_content = 1

        ocr =  tokenize([_preprocess_text(ocr)], context_length=20)[0]
        fake_ocr =  tokenize([_preprocess_text(fake_ocr)], context_length=20)[0]
        eos_index = query.numpy().tolist().index(_tokenizer.vocab['[SEP]'])

        return image, query, ocr, ocr_exists, fake_ocr, ocr_content, eos_index


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
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
    dataset: LMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = LMDBDataset(
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
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs). 
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
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
