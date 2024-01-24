[**中文说明**](README.md) | [**English**](README_EN.md)
<br>

<p align="center">
    <br>
    <img src="assets/logo.svg" width="400" />
<p>

<p align="center">
        <a href="https://github.com/QQBrowserVideoSearch/CBVS-UniCLIP">Code</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2401.10475">Paper</a>
</p>
<br><br>

# 导航
- [导航](#导航)
  - [项目介绍](#项目介绍)
  - [最新发布](#最新发布)
  - [数据集](#数据集)
  - [模型及实验](#模型及实验)
  - [项目结构](#项目结构)
  - [依赖环境](#依赖环境)
  - [使用教程](#使用教程)

# 项目介绍

本项目为填补短视频封面数据的空白，建立了首个针对中文短视频搜索场景的大规模封面文本基准，并提出了用于封面图文对比学习的UniCLIP。本项目代码基于<b>[QA-CLIP](https://github.com/TencentARC-QQ/QA-CLIP)</b>建设，并在<b>[大规模中文短视频封面数据](#数据集)</b>上微调。UniCLIP已被部署到<b>[QQ浏览器](https://browser.qq.com/mobile)</b>在线搜索业务中，并取得了显著的收益。详细的技术细节，请参阅我们的<b>[论文](https://arxiv.org/abs/2401.10475)</b>。
<br>


# 最新发布
* 2024.1.23 [CBVS数据集与UniCLIP项目正式开源](https://github.com/QQBrowserVideoSearch/CBVS-UniCLIP)

# 数据集
CBVS数据集目前可通过<b>[Google Drive](https://drive.google.com/drive/folders/1c4WUzK-s9e2zgcqq_L7jp_CL5Z_YePWh?usp=sharing)</b>或<b>[百度网盘](https://pan.baidu.com/s/1DFP9derk_Twyls4H_o41JQ?pwd=n5uw)</b>下载。CBVS包括三个版本：20K/5M/10M，各版本的具体细节见下表：

<table border="1" width="100%">
    <tr align="center">
        <th>版本</th><th>图文对数量</th><th>有无人工标注</th><th>图像类型</th><th>文本类型</th><th>用途</th>
    </tr>
    <tr align="center">
        <td>CBVS-20K</sub></td><td>20,001</td><td>有</td><td>短视频封面</td><td>用户查询、OCR文本</td><td>测试</td>
    </tr>
    <tr align="center">
        <td>CBVS-5M</sub></td><td>4,767,435</td><td>无</td><td>短视频封面</td><td>短视频标题、OCR文本</td><td>预训练、微调</td>
    </tr>
    <tr align="center">
        <td>CBVS-10M</sub></td><td>10,075,989</td><td>无</td><td>短视频封面</td><td>短视频标题、OCR文本</td><td>预训练、微调</td>
    </tr>
</table>

# 模型及实验
## 模型规模 & 下载链接
UniCLIP目前开源1个规模，其模型信息和下载方式见下表：

<table border="1" width="100%">
    <tr align="center">
        <th>模型规模</th><th>预训练</th><th>微调</th><th>下载链接</th><th>参数量</th><th>视觉侧骨架</th><th>视觉侧参数量</th><th>文本侧骨架</th><th>文本侧参数量</th><th>分辨率</th>
    </tr>
    <tr align="center">
        <td>UniCLIP<sub>ViT-B/16</sub></td><td><a href="https://github.com/TencentARC-QQ/QA-CLIP">QA-CLIP</a></td><td>CBVS-5M</td><td><a href="https://drive.google.com/file/d/1I-aoQ6qaUZ0IrU0o29pBhPTG8LFHLsnJ/view?usp=sharing">Google Drive</a>/<a href="https://pan.baidu.com/s/1EV5BdwnW4FuGWaXRkDQdlA?pwd=7wkp">百度网盘</a></td><td>188M</td><td>ViT-B/16</td><td>86M</td><td>RoBERTa-wwm-Base</td><td>102M</td><td>224</td>
    </tr>
</table>

## 实验结果
我们在CBVS-20K数据集上测试了以下模型的zero-shot与微调结果。

**zero-shot**:
<table border="1" width="100%">
    <tr align="center">
        <th></th><th colspan="4">召回指标</th><th colspan="5">排序指标</th>
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

**微调**:
<table border="1" width="100%">
    <tr align="center">
        <th></th><th colspan="4">召回指标</th><th colspan="5">排序指标</th>
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


# 项目结构

```
CBVS-UniCLIP
├── datasets                     # 数据集存放路径
│   └── cbvs20k.txt              # CBVS-20K数据集，用于测试
├── output                       # 模型检查点与推理结果输出路径
│   └── inference_result.txt     # 示例的模型推理结果
├── pretrained                   # 预训练检查点存放路径
├── run_scripts                  # 训练与推理脚本
│   ├── train_vit_b.sh           # 训练脚本
│   └── eval_vit_b.sh            # 推理脚本
├── uniclip                      # 模型、训练与推理代码
│   ├── clip                     # UniCLIP模型定义
│   ├── eval                     # 推理相关代码
│   └── training                 # 训练相关代码
└── inference.py                 # UniCLIP推理示例
```

# 依赖环境
开始本项目前，需先检查是否满足下列环境配置要求:

* python >= 3.6.4
* pytorch >= 1.8.0 (with torchvision >= 0.9.0)
* CUDA Version >= 10.2

运行下列命令即可安装本项目所需的三方库。

```bash
pip install -r requirements.txt
```


# 使用教程

## 准备工作

### 预训练CKPT

请参考前文[模型及实验](#模型及实验)部分，下载对应模型ckpt。推荐将下载的ckpt文件存放于`CBVS-UniCLIP/pretrained/`目录下。
<br>

### 数据集

请参考前文[数据集](#数据集)部分，下载CBVS数据集。推荐将下载的数据集存放于`CBVS-UniCLIP/datasets/`目录下，推荐的组织结构为：

```
CBVS-UniCLIP
├── datasets                     # 数据集存放路径
│   ├── cbvs20k.txt              # CBVS-20K数据集，用于测试
│   ├── cbvs5m.txt               # CBVS-5M数据集，用于训练
│   ├── cbvs10m.txt              # CBVS-5M数据集，用于训练
│   └── cbvs10m-HNSW             # HNSW算法所需的数据文件
```

我们在数据集中提供图像的URL链接，并在模型数据加载时下载图像。为了提高模型效率，我们推荐预下载图像到本地，从而在模型中加载本地文件。

## 模型finetune

UniCLIP的微调通过如下指令执行：

```bash
cd UniCLIP/
bash run_scripts/train_vit_b.sh
```

相关的训练配置项包括:

+ 分布式
  + `GPUS_PER_NODE`: 每个机器上的GPU个数
  + `WORKER_CNT`: 训练的机器个数
+ 训练数据
  + `train-data`: 训练数据所在路径
  + `num-workers`: 训练集数据处理（DataLoader）的进程数，默认为1
+ 训练超参数
  + `vision-model`: 指定视觉backbone, 从 `["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"]`选择
  + `text-model`: 指定文本backbone, 从 `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`选择
  + `context-length`: 文本输入序列长度
  + `warmup`: warmup步数
  + `batch-size`: 训练时单卡batch-size。（请保证`训练样本总数 > batch-size * GPU数`，至少满足1个训练batch）
  + `lr`: 学习率
  + `wd`: weight decay
  + `max-steps`: 训练步数，也可通过`max-epochs`指定训练轮数
  + `use-augment`: 是否使用[AutoAugment](https://arxiv.org/abs/1805.09501)对图片进行数据增强
  + `accum-freq`: <span id="gradient_accumulation"></span>梯度累积频率，默认为1。指定为大于1的整数时开启对比学习梯度累积，模拟更大的batch size。如果单卡batch size为`m`，则总的batch size为`accum_freq * m * GPU数`
+ 输出选项
  + `output-base-dir`: 指定输出路径
  + `name`: 指定输出子路径。超参日志, 训练日志以及产出ckpt均会存放至 `${output-base-dir}/${name}/`
  + `save-step-frequency`及`save-epoch-frequency`: 存ckpt的步数或轮数间隔
  + `report-training-batch-acc`: 日志是否报告训练图到文&文到图batch准确率
+ 权重读取相关选项
  + `resume`: 权重读取的路径。示例脚本中指定为预训练ckpt路径，也可以指定为用户自己finetune的ckpt路径做继续训练
  + `reset-data-offset`: 是否从此前的数据断点续跑。如batch size或GPU卡数超参改变，建议打开此选项
  + `reset-optimizer`: 是否使用optimizer state
+ 模型选项
  + `ocr-presence`: 是否执行Presence-guided encoder与IC任务
  + `ocr-semantic`: 是否执行Semantic-guided encoder与ITM任务

训练完毕，log 会自动存在`UniCLIP/${output-base-dir}/${name}/out_${timestamp}.log`，训练log格式如下所示:
```
2023-12-15,15:43:23 | INFO | Rank 0 | Global Steps: 514/62740 | Train Epoch: 1 [781280/4768240 (16%)] | Loss: 1.888588 | Loss_ocr_presence: 0.479523 | Loss_ocr_semantic: 0.960589 | Loss_Img_Txt: 2.180721 | Image2Text Acc: 55.59 | Text2Image Acc: 55.33 | OCR_presence_est Acc: 78.42 | OCR_semantic_est Acc: 67.89 | Data Time: 0.042s | Batch Time: 5.787s | LR: 0.000020 | logit_scale: 4.603 | Global Batch Size: 1520
```

**注意**: 对比学习的训练收敛和稳定性和总batch size相关。如您使用更小的batch size（相比默认配置128 per-GPU \* 8 GPU），建议使用更小的学习率。我们推荐使用更多的GPU和更大的batch size以取得更好的效果。

## 预测及评估

### 批量数据预测

UniCLIP的预测通过如下指令执行：

```bash
cd UniCLIP/
bash run_scripts/eval_vit_b.sh
```

相关的推理配置项包括:

+ 推理数据
  + `test-data`: 推理数据所在路径
+ 推理超参数
  + `vision-model`: 指定视觉backbone, 从 `["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"]`选择
  + `text-model`: 指定文本backbone, 从 `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`选择
  + `context-length`: 文本输入序列长度
  + `batch-size`: 推理时单卡batch-size
  + `input-resolution`: 输入图像分辨率，默认为224
+ 输出选项
  + `output-file`: 指定输出路径
+ 权重读取相关选项
  + `resume`: 权重读取的路径
+ 模型选项
  + `ocr-presence`: 是否执行Presence-guided encoder与IC任务
  + `ocr-semantic`: 是否执行Semantic-guided encoder与ITM任务

训练完毕，模型推理结果自动保存在`UniCLIP/{output-file}`，标签、查询、图像ID、模型打分以`\t`作为分隔符，如下所示:
```
1	大宅门演员表	70001002_d3349orxhl9	0.3051548898220062
2	大宅门演员表	70001002_h0871hjdde8	0.39615771174430847
2	明日战记在线观看	70001002_e3348lunfza	0.37629106640815735
```

与此同时，脚本将计算并输出模型的排序指标，包括PNR、NDCG与MAP。

**注意**: 推理与训练脚本中的推理配置项需保持一致。


### 单条数据预测

我们还提供了使用UniCLIP预测单条数据的代码示例，见`inference.py`。可通过修改下面的代码，执行图(url)文(query)相关性推理。

```python
score = compute_feature(model, query='大宅门演员表', url='http://puui.qpic.cn/vpic_cover/h0871hjdde8/h0871hjdde8_hz.jpg/640', compute_score=True)
```


# 引用

如果我们的工作对您有帮助，欢迎引用我们的工作：

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