[**English**](README_EN.md) | [**中文说明**](README.md)
<br>

<p align="center">
    <br>
    <img src="assets/logo.svg" width="350" />
<p>

<p align="center">
        <a href="https://github.com/QQBrowserVideoSearch/CBVS-UniCLIP">Code</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2401.10475">Paper</a>
</p>

# Catalogues
- [Catalogues](#Catalogues)
  - [Introduction](#Introduction)
  - [News](#News)
  - [Dataset](#Dataset)
  - [Models](#Models)
  - [Structure](#Structure)
  - [Environment](#Environment)
  - [Tutorials](#Tutorials)

# Introduction
In this project, to fill the gap of short video cover data, we establish the first large-scale cover image-text benchmark for Chinese short video search scenarios, and propose UniCLIP for image-text contrastive learning. This project code is built based on <b>[QA-CLIP](https://github.com/TencentARC-QQ/QA-CLIP)</b> and fine-tuned on l<b>[arge-scale Chinese short video cover data](#Dataset)</b>. UniCLIP has been deployed to <b>[QQ Browser](https://browser.qq.com/mobile)</b> online search business and achieved significant benefits. For more technical details, please refer to our <b>[paper](https://arxiv.org/abs/2401.10475)</b>.
<br>

# News
* 2024.1.23 [CBVS dataset and UniCLIP project officially open source](https://github.com/QQBrowserVideoSearch/CBVS-UniCLIP)

# Dataset
The CBVS dataset is currently available for download via <b>[Google Drive](https://drive.google.com/drive/folders/1c4WUzK-s9e2zgcqq_L7jp_CL5Z_YePWh?usp=sharing)</b> or <b>[Baidu online disk](https://pan.baidu.com/s/1DFP9derk_Twyls4H_o41JQ?pwd=n5uw)</b>. CBVS consists of three versions: 20K/5M/10M, and the specific details of each version are shown in the table below:

<table border="1" width="100%">
    <tr align="center">
        <th>Version</th><th>#Pairs</th><th>Annotations</th><th>Image type</th><th>text type</th><th>Purpose</th>
    </tr>
    <tr align="center">
        <td>CBVS-20K</sub></td><td>20,001</td><td>T</td><td>Short video cover</td><td>User queries, OCR text</td><td>Test</td>
    </tr>
    <tr align="center">
        <td>CBVS-5M</sub></td><td>4,767,435</td><td>F</td><td>Short video cover</td><td>Short video titles, OCR text</td><td>Pre-training, fine-tuning</td>
    </tr>
    <tr align="center">
        <td>CBVS-10M</sub></td><td>10,075,989</td><td>F</td><td>Short video cover</td><td>Short video titles, OCR text</td><td>Pre-training, fine-tuning</td>
    </tr>
</table>

# Models
## Model Scale & Download Links
UniCLIP is currently open-sourced at 1 scale, and its model information and download methods are shown in the table below:

<table border="1" width="100%">
    <tr align="center">
        <th>Model scale</th><th>Pre-training</th><th>Fine-tuning</th><th>Download</th><th>#Params</th><th>Vision side</th><th>#Vision params</th><th>Text side</th><th>#text params</th><th>Resolution</th>
    </tr>
    <tr align="center">
        <td>UniCLIP<sub>ViT-B/16</sub></td><td><a href="https://github.com/TencentARC-QQ/QA-CLIP">QA-CLIP</a></td><td>CBVS-5M</td><td><a href="https://drive.google.com/file/d/1I-aoQ6qaUZ0IrU0o29pBhPTG8LFHLsnJ/view?usp=sharing">Google Drive</a>/<a href="https://pan.baidu.com/s/1EV5BdwnW4FuGWaXRkDQdlA?pwd=7wkp">Baidu online disk</a></td><td>188M</td><td>ViT-B/16</td><td>86M</td><td>RoBERTa-wwm-Base</td><td>102M</td><td>224</td>
    </tr>
</table>

## Results
We test the following models on the CBVS-20K dataset for zero-shot and fine-tuning results.

**zero-shot**:
<table border="1" width="100%">
    <tr align="center">
        <th></th><th colspan="4">Recall metrics</th><th colspan="5">Rank metrics</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td><td>PNR</td><td>NDCG@1</td><td>NDCG@5</td><td>NDCG@10</td><td>MAP</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT−B/16</sub></td><td>0.384</td><td>0.628</td><td>0.704</td><td>0.572</td><td>2.718</td><td>0.768</td><td>0.835</td><td>0.885</td><td>0.764</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT−L/14</sub></td><td>0.434</td><td>0.685</td><td>0.756</td><td>0.625</td><td>2.812</td><td>0.773</td><td>0.840</td><td>0.889</td><td>0.775</td>
    </tr>
    <tr align="center">
        <td width="120%">Wukong<sub>ViT−B/32</sub></td><td>0.197</td><td>0.356</td><td>0.439</td><td>0.331</td><td>2.000</td><td>0.702</td><td>0.791</td><td>0.858</td><td>0.712</td>
    </tr>
    <tr align="center">
        <td width="120%">Wukong<sub>ViT−L/14</sub></td><td>0.311</td><td>0.503</td><td>0.583</td><td>0.466</td><td>2.229</td><td>0.739</td><td>0.811</td><td>0.872</td><td>0.738</td>
    </tr>
    <tr align="center">
        <td width="120%">Taiyi-CLIP<sub>ViT−B</sub></td><td>0.251</td><td>0.445</td><td>0.525</td><td>0.407</td><td>2.142</td><td>0.718</td><td>0.800</td><td>0.861</td><td>0.727</td>
    </tr>
    <tr align="center">
        <td width="120%">Taiyi-CLIP<sub>ViT−L</sub></td><td>0.269</td><td>0.492</td><td>0.577</td><td>0.446</td><td>2.278</td><td>0.726</td><td>0.805</td><td>0.866</td><td>0.733</td>
    </tr>
    <tr align="center">
        <td width="120%">Ernie-ViL2.0<sub>ViT−B</sub></td><td>0.413</td><td>0.660</td><td>0.742</td><td>0.605</td><td>2.759</td><td>0.764</td><td>0.835</td><td>0.886</td><td>0.768</td>
    </tr>
    <tr align="center">
        <td width="120%">R2D2-23M<sub>ViT−L/14</sub></td><td>0.258</td><td>0.407</td><td>0.436</td><td>0.367</td><td>2.312</td><td>0.733</td><td>0.810</td><td>0.868</td><td>0.738</td>
    </tr>
    <tr align="center">
        <td width="120%">R2D2-250M<sub>ViT−L/14</sub></td><td>0.356</td><td>0.512</td><td>0.535</td><td>0.468</td><td>2.829</td><td>0.789</td><td>0.842</td><td>0.891</td><td>0.775</td>
    </tr>
    <tr align="center">
        <td width="120%">AltCLIP<sub>ViT−L</sub></td><td>0.162</td><td>0.284</td><td>0.336</td><td>0.261</td><td>1.869</td><td>0.669</td><td>0.771</td><td>0.842</td><td>0.701</td>
    </tr>
    <tr align="center">
        <td width="120%">QA-CLIP<sub>ViT−B/16</sub></td><td>0.400</td><td>0.652</td><td>0.724</td><td>0.592</td><td>2.804</td><td>0.774</td><td>0.838</td><td>0.888</td><td>0.770</td>
    </tr>

</table>

**fine-tuning**:
<table border="1" width="100%">
    <tr align="center">
        <th></th><th colspan="4">Recall metrics</th><th colspan="5">Rank metrics</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td><td>PNR</td><td>NDCG@1</td><td>NDCG@5</td><td>NDCG@10</td><td>MAP</td>
    </tr>
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT−B/16</sub></td><td>0.471</td><td>0.721</td><td>0.796</td><td>0.663</td><td>2.914</td><td>0.767</td><td>0.838</td><td>0.888</td><td>0.767</td>
    </tr>
    <tr align="center">
        <td width="120%">R2D2-250M<sub>ViT−L/14</sub></td><td>0.418</td><td>0.605</td><td>0.650</td><td>0.558</td><td>2.934</td><td>0.780</td><td>0.844</td><td>0.891</td><td>0.774</td>
    </tr>
    <tr align="center">
        <td width="120%">QA-CLIP<sub>ViT−B/16</sub></td><td>0.473</td><td>0.711</td><td>0.783</td><td>0.656</td><td>2.907</td><td>0.778</td><td>0.841</td><td>0.890</td><td>0.771</td>
    </tr>
    <tr align="center">
        <td width="120%">UniCLIP<sub>ViT−B/16</sub></td><td><b>0.503</b></td><td><b>0.754</b></td><td><b>0.820</b></td><td><b>0.692</b></td><td><b>3.069</b></td><td>0.784</td><td><b>0.846</b></td><td><b>0.893</b></td><td><b>0.779</b></td>
    </tr>
</table>


# Structure

```
CBVS-UniCLIP
├── datasets                     # Dataset storage path
│   └── cbvs20k.txt              # CBVS-20K dataset for testing
├── output                       # Model checkpoints and inference result output path
│   └── inference_result.txt     # Example model inference results
├── pretrained                   # Pre-training checkpoint storage path
├── run_scripts                  # Training and reasoning scripts
│   ├── train_vit_b.sh           # Training script
│   └── eval_vit_b.sh            # Reasoning script
├── uniclip                      # Model, training and inference code
│   ├── clip                     # UniCLIP model definition
│   ├── eval                     # Reasoning related codes
│   └── training                 # Training related codes
└── inference.py                 # Examples of UniCLIP reasoning
```

# Environment
Before starting this project, check that the following environment configuration requirements are met:

* python >= 3.6.4
* pytorch >= 1.8.0 (with torchvision >= 0.9.0)
* CUDA Version >= 10.2

Run the following commands to install the libraries required for this project.

```bash
pip install -r requirements.txt
```


# Tutorials

## Preliminary

### Pre-training CKPT

Please refer to the previous section [Models](#Models) to download the corresponding model ckpt. It is recommended to store the downloaded ckpt file in the `CBVS-UniCLIP/pretrained/` directory.
<br>

### Dataset

Please refer to the previous section [Dataset](#Dataset) to download the CBVS dataset. It is recommended to store the downloaded dataset file in the `CBVS-UniCLIP/datasets/` directory. The recommended organisational structure is:

```
CBVS-UniCLIP
├── datasets                     # Dataset storage path
│   ├── cbvs20k.txt              # CBVS-20K dataset for testing
│   ├── cbvs5m.txt               # CBVS-5M dataset for training
│   ├── cbvs10m.txt              # CBVS-10M dataset for training
│   └── cbvs10m-HNSW             # Data files required for the HNSW algorithm
```

We provide URL links to the images in the dataset and download the images when the model data is loaded. To improve model efficiency, we recommend pre-downloading images locally so that local files are loaded in the model.

## Finetune of UniCLIP

Finetuning of UniCLIP is performed by the following instructions:

```bash
cd UniCLIP/
bash run_scripts/train_vit_b.sh
```

Relevant training configuration items include:

+ Distributed
  + `GPUS_PER_NODE`: Number of GPUs per machine
  + `WORKER_CNT`: Number of machines
+ Data
  + `train-data`: Path to data
  + `num-workers`: Number of processes for training set data processing (DataLoader), default is 1
+ Training hyperparameters
  + `vision-model`: Specify vision backbone, select from `["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"]`
  + `text-model`: Specify text backbone, select from `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`
  + `context-length`: Text input sequence length
  + `warmup`: Warmup steps
  + `batch-size`: Single card batch-size for training. (Please ensure that `total number of training samples > batch-size * number of GPUs` is satisfied for at least 1 training batch)
  + `lr`: learning rate
  + `wd`: weight decay
  + `max-steps`: The number of training steps, or the number of training rounds can be specified via `max-epochs`.
  + `use-augment`: Whether or not to use [AutoAugment](https://arxiv.org/abs/1805.09501) for data augmentation of images
  + `accum-freq`: <span id="gradient_accumulation"></span>Gradient Accumulation Frequency, default is 1. Specify an integer greater than 1 to turn on contrast learning gradient accumulation and simulate a larger batch size. if the single card batch size is `m`, the total batch size is `accum_freq * m * number of GPUs`
+ Output options
  + `output-base-dir`: Specify the output path
  + `name`: Specify the output subpath. The hyperparameter log, training log and output ckpt will be stored in `${output-base-dir}/${name}/`.
  + `save-step-frequency` and`save-epoch-frequency`: The number of steps or round intervals for storing a ckpt
  + `report-training-batch-acc`: Whether logs report training image-to-text & text-to-image batch accuracy
+ Weight reading related options
  + `resume`: The path for weight reading. The example script specifies the pre-training ckpt path, or you can specify the user's own finetune ckpt path for further training
  + `reset-data-offset`: Whether or not to break the run from the previous data. It is recommended to turn this option on if the batch size or GPU cardinality has changed.
  + `reset-optimizer`: Whether or not to use optimizer state
+ Model options
  + `ocr-presence`: Whether or not to perform Presence-guided encoder with IC tasks
  + `ocr-semantic`: Whether or not to perform Semantic-guided encoder with ITM tasks

After training, the log is automatically saved in `UniCLIP/${output-base-dir}/${name}/out_${timestamp}.log`, and the format of the training log is shown as follows.

```
2023-12-15,15:43:23 | INFO | Rank 0 | Global Steps: 514/62740 | Train Epoch: 1 [781280/4768240 (16%)] | Loss: 1.888588 | Loss_ocr_presence: 0.479523 | Loss_ocr_semantic: 0.960589 | Loss_Img_Txt: 2.180721 | Image2Text Acc: 55.59 | Text2Image Acc: 55.33 | OCR_presence_est Acc: 78.42 | OCR_semantic_est Acc: 67.89 | Data Time: 0.042s | Batch Time: 5.787s | LR: 0.000020 | logit_scale: 4.603 | Global Batch Size: 1520
```

**Note**: The training convergence and stability of contrast learning is related to the total batch size. If you use a smaller batch size (128 per-GPU \* 8 GPUs compared to the default configuration), it is recommended to use a smaller learning rate. We recommend using more GPUs and a larger batch size for better results.

## Prediction

### Batch data prediction

The UniCLIP prediction is executed by the following instruction:

```bash
cd UniCLIP/
bash run_scripts/eval_vit_b.sh
```

Relevant reasoning configuration items include:

+ Data
  + `test-data`: Path to data
+ Inference hyperparameters
  + `vision-model`: Specify vision backbone, select from `["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"]`
  + `text-model`: Specify text backbone, select from `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`
  + `context-length`: Text input sequence length
  + `batch-size`: Single-card batch-size for inference
  + `input-resolution`: Input image resolution, default is 224
+ Output options
  + `output-file`: Specify the output path
+ Weight reading related options
  + `resume`: Path for weight reading
+ Model options
  + `ocr-presence`: Whether or not to perform Presence-guided encoder with IC tasks
  + `ocr-semantic`: Whether or not to perform Semantic-guided encoder with ITM tasks

After training, the model inference results are automatically saved in `UniCLIP/{output-file}`, with label, query, image ID, and model scoring separated by `\t` as follows.

```
1	大宅门演员表	70001002_d3349orxhl9	0.3051548898220062
2	大宅门演员表	70001002_h0871hjdde8	0.39615771174430847
2	明日战记在线观看	70001002_e3348lunfza	0.37629106640815735
```

At the same time, the script will calculate and output the model's ranking metrics, including PNR, NDCG & MAP.


**Note**: The reasoning needs to be consistent with the configuration items in the training script.


### Individual data prediction

We also provide code examples for predicting a single piece of data using UniCLIP, see `inference.py`. Image (url) text (query) correlation inference can be performed by modifying the code below.

```python
score = compute_feature(model, query='大宅门演员表', url='http://puui.qpic.cn/vpic_cover/h0871hjdde8/h0871hjdde8_hz.jpg/640', compute_score=True)
```


# Cite

Please cite our work if it has been helpful to you:

```
@misc{qiao2024cbvs,
      title={CBVS: A Large-Scale Chinese Image-Text Benchmark for Real-World Short Video Search Scenarios}, 
      author={Xiangshuo Qiao and Xianxin Li and Xiaozhe Qu and Jie Zhang and Yang Liu and Yu Luo and Cihang Jin and Jin Ma},
      year={2024},
      eprint={2401.10475},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```