<!---
Copyright 2022 The OFA-Sys Team. 
All rights reserved.
This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory.
-->

<p align="center">
    <br>
    <img src="examples/OFA_logo_tp_path.svg" width="150" />
    <br>
<p>
<br>

<p align="center">
        <a href="modelscope.md">ModelScope</a>&nbsp ｜ &nbsp<a href="checkpoints.md">Checkpoints</a>&nbsp ｜ &nbsp<a href="colab.md">Colab</a>&nbsp ｜ &nbsp<a href="https://huggingface.co/ofa-sys">Demo</a>&nbsp ｜ &nbsp<a href="http://arxiv.org/abs/2202.03052">Paper </a>&nbsp ｜ &nbspBlog
</p>

<p align="center">
    <br>
    <img src="examples/demo.gif" width="800" />
    <br>
<p>

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>

OFA is a unified sequence-to-sequence pretrained model (support **English** and **Chinese**) that unifies modalities (i.e., cross-modality, vision, language) and tasks (**finetuning** and **prompt tuning** are supported): image captioning (1st at the [MSCOCO Leaderboard](https://competitions.codalab.org/competitions/3221#results)), VQA ([link](https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278)), visual grounding, text-to-image generation, text classification, text generation, image classification, etc. We provide **step-by-step** instructions for pretraining and finetuning and corresponding checkpoints (check official ckpt \[[EN](checkpoints.md)|[CN](checkpoints_cn.md)\] or [huggingface ckpt](https://huggingface.co/OFA-Sys)).

We sincerely welcome contributions to our project. Feel free to contact us or send us issues / PRs!
<br></br>

# Online Demos
We provide online demo via Hugging Face Spaces for you to interact with our pretrained and finetuned models. Below are the links to the demos:
* Image Captioning \[[ModelScope](https://modelscope.cn/#/models/damo/ofa_image-caption_coco_large_en/summary)  |  [Spaces](https://huggingface.co/spaces/OFA-Sys/OFA-Image_Caption)\]
* Visual Grounding \[[ModelScope](https://modelscope.cn/#/models/damo/ofa_visual-grounding_refcoco_large_en/summary) | [Spaces](https://huggingface.co/spaces/OFA-Sys/OFA-Visual_Grounding)\]
* Visual Question Answering \[[ModelScope](https://modelscope.cn/#/models/damo/ofa_visual-question-answering_pretrain_large_en/summary) | [Spaces](https://huggingface.co/spaces/OFA-Sys/OFA-Visual_Question_Answering)\]
* Text-to-Image Generation \[[ModelScope](https://modelscope.cn/#/models/damo/ofa_text-to-image-synthesis_coco_large_en/summary) | [Spaces](https://huggingface.co/spaces/OFA-Sys/OFA-Text2Image_Generation)\]
* Generic Interface \[[Spaces](https://huggingface.co/spaces/OFA-Sys/OFA-Generic_Interface)\]

Also we provide Colab notebooks for you to better perceive the procedures. Click [here](colab.md) to check them out!
<br></br>

# Requirements
* python 3.7.4
* pytorch 1.8.1
* torchvision 0.9.1
* JAVA 1.8 (for COCO evaluation)
<br></br>

# Installation
```bash
git clone https://github.com/OFA-Sys/OFA
pip install -r requirements.txt
```
<br></br>

# Datasets and Checkpoints
See [datasets.md](datasets.md) and [checkpoints.md](checkpoints.md).
<br></br>

# Training & Inference
Below we provide methods for training and inference on different tasks. We provide both pretrained OFA-Large and OFA-Base in [checkpoints.md](checkpoints.md). The scripts mentioned in this section are prepared for OFA-Large. For reproducing the downstreaming results of OFA-Base, we have also provided the corresponding finetuning and inference scripts for OFA-Base in the `run_scripts/` folder.

We recommend that your workspace directory should be organized like this: 
```
OFA/
├── checkpoints/
│   ├── ofa_base.pt
│   ├── ofa_large.pt
│   ├── caption_large_best_clean.pt
│   └── ...
├── criterions/
├── data/
├── dataset/
│   ├── caption_data/
│   ├── gigaword_data/
│   └── ...
├── fairseq/
├── models/
├── run_scripts/
├── tasks/
├── train.py
├── trainer.py
└── utils/
```

## Image Processing
To ensure the efficiency of processing data, we did not store images with small files, but instead we encode them to base64 strings.
Transforming image files to base64 strings is simple. Run the following code:
```python
from PIL import Image
from io import BytesIO
import base64

img = Image.open(file_name) # path to file
img_buffer = BytesIO()
img.save(img_buffer, format=img.format)
byte_data = img_buffer.getvalue()
base64_str = base64.b64encode(byte_data) # bytes
base64_str = base64_str.decode("utf-8") # str
```

## Pretraining
Below we provide methods for pretraining OFA.

<details>
    <summary><b>1. Prepare the Dataset</b></summary>
    <p>
        To pretrain OFA, you should first download the dataset we provide (<a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/pretrain_data/pretrain_data_examples.zip">pretrain_data_examples.zip</a>, a small subset of the original pretraining data). For your customed pretraining datasets, please prepare your training samples into the same format. <code>pretrain_data_examples.zip</code> contains 4 TSV files: <code>vision_language_examples.tsv</code>, <code>text_examples.tsv</code>, <code>image_examples.tsv</code> and <code>detection_examples.tsv</code>. Details of these files are as follows: 
        <br />
        <ul type="circle">
            <li><b>vision_language_examples.tsv</b>:
    Each line contains uniq-id, image (base64 string), caption, question, answer, ground-truth objects (objects appearing in the caption or question), dataset name (source of the data) and task type (caption, qa or visual gronunding). Prepared for the pretraining tasks of visual grounding, grounded captioning, image-text matching, image captioning and visual question answering. </li>
            <li><b>text_examples.tsv</b>: Each line contains uniq-id and text. Prepared for the pretraining task of text infilling. </li> 
            <li><b>image_examples.tsv</b>: Each line contains uniq-id, image (base64 string, should be resized to 256*256 resolution) and image-code (generate the sparse codes for the central part of image through VQ-GAN). Prepared for the pretraining task of image infilling. </li>
            <li><b>detection_examples.tsv</b>: Each line contains uniq-id, image (base64 string) and bounding box annotations (contains the top-left and bottom-right coordinates of the bounding box, object_id and object_name, seperated by commas). Prepared for the pretraining task of detection. </li>
        </ul>
        In addition, the folder negative_sample in pretrain_data_examples.zip contains three files <code>all_captions.txt</code>, <code>object.txt</code> and <code>type2ans.json</code>. The data in these files are used as negative samples for the image-text matching (ITM) task.
    </p>
</details>
<details>
    <summary><b>2. Pretraining</b></summary>
    <p>
        By default, the pretraining script will attempt to restore the released pretrained checkpoints of OFA-Base or OFA-Large and perform continuous pretraining. Continuous pretraining is more recommended, which achieves much better results compared with pretraining from scratch. For continuous pretraining, please download the pretrained weights in advance (see <a href='checkpoints.md'>checkpoints.md</a>) and put them in the correct directory <code>OFA/checkpoints/</code>. If not, the pretraining will begin from scratch.
    </p>
<pre>
cd run_scripts/pretraining
bash pretrain_ofa_large.sh # Pretrain OFA-Large. For OFA-Base, use pretrain_ofa_base.sh
</pre>
    <p>
        If the pretrained OFA checkpoint is restored successfully, you will see the following information in the log:
    </p>
<pre>
INFO: Loaded checkpoint ../../checkpoints/ofa_large.pt
</pre>
</details>

## Visual Grounding (Referring Expression Comprehension)
Here provides procedures for you to prepare data, train, and evaluate your model on visual grounding. 
<details>
    <summary><b>1. Prepare the Dataset & Checkpoints</b></summary>
    <p>
        Download data (see <a href='datasets.md'>datasets.md</a>) and models (see <a href='checkpoints.md'>checkpoints.md</a>) and put them in the correct directory. We provide RefCOCO (split by UNC), RefCOCO+ (split by UNC) and RefCOCOg (split by UMD) datasets. See <a href='https://www.tensorflow.org/datasets/catalog/ref_coco'>RefCOCO</a> and <a href="https://github.com/lichengunc/refer">Refer</a> for more details. Note that in the original dataset, each region-coord (or bounding box) may corresponds to multiple descriptive texts. We split these texts into multiple samples so that the region-coord in each sample corresponds to only one text. Each line of the processed dataset represents a sample with the following format. The information of uniq-id, image-id, text, region-coord (separated by commas), image base64 string are separated by tabs.
    </p>
<pre>
79_1    237367  A woman in a white blouse holding a glass of wine.  230.79,121.75,423.66,463.06 9j/4AAQ...1pAz/9k=
</pre>
</details>
<details>
    <summary><b>2. Finetuning</b></summary>
    <p>
        Unlike the original paper, we finetune OFA with a drop-path rate of 0.2, and found that training with this hyper-parameter achieves better results. We will update the reported results of the paper later.
    </p>
<pre>
cd run_scripts/refcoco
nohup sh train_refcoco.sh > train_refcoco.out &  # finetune for refcoco
nohup sh train_refcocoplus.sh > train_refcocoplus.out &  # finetune for refcoco+
nohup sh train_refcocog.sh > train_refcocog.out &  # finetune for refcocog
</pre>
</details>
<details>
    <summary><b>3. Inference</b></summary>
    <p>
        Run the following commands for the evaluation. 
    </p>
<pre>
cd run_scripts/refcoco ; sh evaluate_refcoco.sh  # inference & evaluate for refcoco/refcoco+/refcocog
</pre>
</details>

