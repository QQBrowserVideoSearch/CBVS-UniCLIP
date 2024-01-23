import torch
import uniclip.clip as clip
from uniclip.clip import image_transform, available_models
from uniclip.clip.model_eval import convert_weights, CLIP
from pathlib import Path
from PIL import Image
import os
import json
import numpy as np
import time
from tqdm import tqdm
import requests
from PIL import Image
from PIL import ImageFile
from io import BytesIO
from uniclip.eval.data import get_eval_dataset
from uniclip.eval.params import parse_args
from uniclip.eval.utils import get_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def load_model(ocr_presence, ocr_semantic, model_path=None, device="cuda", vision_model=None, text_model=None, input_resolution=224):
    # Initialize the model.
    vision_model_config_file = f"uniclip/clip/model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = f"uniclip/clip/model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers']) 

        for k, v in json.load(ft).items():
            model_info[k] = v
    print('Model info', model_info)
    
    model = CLIP(**model_info, ocr_presence=ocr_presence, ocr_semantic=ocr_semantic)
    convert_weights(model)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        sd = {}
        for key in checkpoint["state_dict"]:
            if 'exist' in key:
                sd[key.replace('exist', 'presence')] = checkpoint["state_dict"][key]
            elif 'content' in key:
                sd[key.replace('content', 'semantic')] = checkpoint["state_dict"][key]
            else:
                sd[key] = checkpoint["state_dict"][key]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}

        model.load_state_dict(sd)
        print(
            f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)"
        )
    return model.to(device), image_transform(input_resolution)

import numpy as np


if __name__ == '__main__':
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(ocr_presence=args.ocr_presence, ocr_semantic=args.ocr_semantic, model_path=args.resume, device=device, 
                                   vision_model=args.vision_model, text_model=args.text_model, 
                                   input_resolution=args.input_resolution)
    model.eval()
    model.float()
    eval_data = get_eval_dataset(args.test_data, args.batch_size)

    with open(args.output_file, 'w') as fout:
        dataloader = eval_data.dataloader
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # text_ids:label \t query \t docid
                text_ids, imgs, texts = batch
                
                imgs = imgs.cuda(0, non_blocking=True)
                texts = texts.cuda(0, non_blocking=True)

                image_features, text_features = model(imgs, texts)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                score = image_features @ text_features.t()

                for idx, text_id in enumerate(text_ids):
                    fout.write(f'{text_id}\t{score[idx][idx].item()}\n')

    get_metrics(args.output_file)
