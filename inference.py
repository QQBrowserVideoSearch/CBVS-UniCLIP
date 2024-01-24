import torch
import uniclip.clip as clip
from uniclip.clip import image_transform, available_models
from uniclip.clip.model import convert_weights, CLIP
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

preprocess = image_transform(224)

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

def get_img_online(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def compute_feature(model, query, url, compute_score=True):
    score = 0
    image_features = None
    text_features = None

    image = get_img_online(url)
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(query, context_length=50).to(device)

    with torch.no_grad():
        image_features = model(image, None)
        text_features = model(None, text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    if compute_score:
        score = image_features @ text_features.t()
        score = score.item()
    
    return score


if __name__ == '__main__':
    # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
    print("Available models:", available_models())  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:{device}")

    model_path = 'pretrained/UniCLIP-base-QA-CLIP-5M.pt'

    model, preprocess = load_model(ocr_presence=1, ocr_semantic=1, model_path=model_path, device=device, 
                                   vision_model="ViT-B-16",text_model="RoBERTa-wwm-ext-base-chinese", 
                                   input_resolution=224)
    model.eval()
    model.float()

    score = compute_feature(model, query='大宅门演员表', url='http://puui.qpic.cn/vpic_cover/h0871hjdde8/h0871hjdde8_hz.jpg/640', compute_score=True)
    print('score:', score)