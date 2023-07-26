# LTrack: Generalizing Multiple Object Tracking to Unseen Domains by Introducing Natural Language Representation

This repository is an official implementation of the AAAI-2023 accepted paper [LTrack: Generalizing Multiple Object Tracking to Unseen Domains by Introducing Natural Language Representation](https://arxiv.org/pdf/2212.01568.pdf).

## Introduction

**TL; DR.** LTrack is a fully end-to-end multiple-object tracking framework based on Transformer. It introduces natural language representaion from vision-language model [CLIP](https://arxiv.org/pdf/2105.03247.pdf) to the MOT tracker for the first time. We hope this work can shed light on how to develop MOT trackers with promising generalization ability to some extent by combining the knowledge from image and language. 

<div style="align: center">
<img src=./figs/LTrack.png/>
</div>

**Abstract.** Although existing multi-object tracking (MOT) algorithms have obtained competitive performance on various benchmarks, almost all of them train and validate models on the same domain. The domain generalization problem of MOT is hardly studied. To bridge this gap, we first draw the observation that the high-level information contained in natural language is domain invariant to different tracking domains. Based on this observation, we propose to introduce natural language representation into visual MOT models for boosting the domain generalization ability. However, it is infeasible to label every tracking target with a textual description. To tackle this problem, we design two modules, namely visual context prompting (VCP) and visual-language mixing (VLM). Specifically, VCP generates visual prompts based on the input frames. VLM joints the information in the generated visual prompts and the textual prompts from a pre-defined Trackbook to obtain instance-level pseudo textual description, which is domain invariant to different tracking scenes. Through training models on MOT17 and validating them on MOT20, we observe that the pseudo textual descriptions generated by our proposed modules improve the generalization performance of query-based trackers by large margins.

## Updates
- (2023/07/26) Code is released.

## Main Results

### MOT17

| **Method** | **Dataset** |    **Train Data**    | **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** | **IDS** |                                           **URL**                                           |
| :--------: | :---------: | :------------------: | :------: | :------: | :------: | :------: | :------: | :-----: | :-----------------------------------------------------------------------------------------: |
|    LTrack    |    MOT17    | MOT17+CrowdHuman Val |   57.5   |   59.4   |   56.1   |   72.1   |   69.1   |  2100   | [model](https://drive.google.com/file/d/1rS_HRr-3oey_DHQa8n9_zixKahWQJ6jl/view?usp=drive_link) |

### DanceTrack

| **Method** | **Dataset** | **Train Data** | **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :--------: | :---------: | :------------: | :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|    LTrack    | MOT20  |   MOT17+CrowdHuman Val   |   46.8   |   45.4   |   48.4   |   57.8   |   61.1   | [model](https://drive.google.com/file/d/1rS_HRr-3oey_DHQa8n9_zixKahWQJ6jl/view?usp=drive_link) |

### BDD100K

| **Method** | **Dataset** | **Train Data** | **HOTA-p** | **AssA-p** | **IDF1-p** |                                           **URL**                                           |
| :--------: | :---------: | :------------: | :------: | :------: | :-----: | :-----------------------------------------------------------------------------------------: |
|    MOTR    |   BDD100K   |    MOT17+CrowdHuman Val     |   33.7   |   39.3   |  40.6   | [model](https://drive.google.com/file/d/1rS_HRr-3oey_DHQa8n9_zixKahWQJ6jl/view?usp=drive_link) |

*Note:*

1. LTrack on MOT17 and Crowdhuman is trained on 8 NVIDIA TESLA V100 GPUs.
2. The training time for MOT17 is about 2.5 days on V100;
3. The inference speed is about 7.0 FPS for resolution 1536x800;
4. All models of LTrack are trained with ResNet50 with pre-trained weights on COCO dataset.


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). We use the [CLIP](https://github.com/openai/CLIP) text encoder to extract language embedding.

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```
### Pre-trained CLIP Models
* Download the pre-trained CLIP models (We use CLIP [RN50.pt](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt)) and save them to the pre_trained folder.

## Usage

### Dataset preparation

1. Please download [MOT17 dataset](https://motchallenge.net/) and [CrowdHuman dataset](https://www.crowdhuman.org/) and organize them like [FairMOT](https://github.com/ifzhang/FairMOT) as following:

```
.
├── crowdhuman
│   ├── images
│   └── labels_with_ids
├── MOT17
│   ├── images
│   ├── labels_with_ids
├── MOT20
│   ├── images
│   ├── labels_with_ids
├── bdd100k
│   ├── images
│       ├── track
│           ├── train
│           ├── val
│   ├── labels
│       ├── track
│           ├── train
│           ├── val

```

2. For BDD100K dataset, you can use the following script to generate txt file:


```bash 
cd datasets/data_path
python3 generate_bdd100k_mot.py
cd ../../
```

### Training and Evaluation

#### Training on single node

You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). Then training MOTR on 8 GPUs as following:

```bash 
sh configs/r50_clip_motr_train.sh

```

#### Evaluation on MOT17

You can download the pretrained model of MOTR (the link is in "Main Results" session), then run following command to evaluate it on MOT17 test dataset (submit to server):

```bash
sh configs/r50_motr_submit_mot17.sh

```
#### Evaluation on MOT20
```bash
sh configs/r50_motr_eval_mot20.sh

```


#### Evaluation on BDD100K

```bash
sh configs/r50_motr_eval_bdd100k.sh

```




## Citing LTrack
If you find LTrack useful in your research, please consider citing:
```bibtex
@inproceedings{yu2023generalizing,
  title={Generalizing multiple object tracking to unseen domains by introducing natural language representation},
  author={Yu, En and Liu, Songtao and Li, Zhuoling and Yang, Jinrong and Li, Zeming and Han, Shoudong and Tao, Wenbing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={3},
  pages={3304--3312},
  year={2023}
}
```
