# SpatialMix and HardPatchesMining for B2MA in Semantic Segmentation

## Abstract

```tex
Aiming at domain adaptive semantic segmentation, conventional unsupervised domain adaptation (UDA) methods 
minimize the discrepancy between source and target data, which requires the coexistence of both data domains. 
Although source-free model adaptation (SFMA) methods avoid access to source data, 
they are dependent on the weights of the pre-trained model, 
which may still leak the source distribution and increase the overhead of model transmission. 
This study identifies a challenging but practical UDA setting for semantic segmentation, 
namely Black-Box Model Adaptation (B2MA). 
The B2MA setting forbids accessing either source data or the pre-trained model weights. 
Instead, it only provides pseudo labels for the given images. 
To avoid overfitting noisy pseudo labels and mitigate the class imbalance issue, 
we cast the B2MA problem as two-stage knowledge distillation: 
1) during cross-domain distillation, 
noisy pseudo labels supervise the target model to imitate the prediction of the source model; 
2) during self distillation, 
a novel data augmentation strategy, namely SpatialMix, 
is introduced to prevent self-distillation from overfitting noisy pseudo labels. 
Meanwhile, a Hard Patches Mining mechanism is designed to increase the contribution of long-tailed classes, 
which alleviates the class imbalance issue. 
Experimental results confirm that 
the B2MA problem can be well tackled via the proposed two-stage distillation, 
which even outperforms the leading SFMA baselines.
```

## Installation

> the first two steps can be skiped if the cityscapes dataset already exists

- download Cityscapes dataset from the official website

- unzip datasets in /path/to/cityscapes/leftImg8bit and /path/to/cityscapes/gtFine

- transfer the GT labels into format used for training

```bash
python ./utils/transfer_trainid.py \
--dataset cityscapes \
--ori-label-root /path/to/cityscapes/gtFine \
--dst-label-root /path/to/cityscapes/gtFine2
```

- set soft links to the datasets

## Testing

## Training

- download the pretrained source-only models of AdaptSegNet

```bash
mkdir pretrained && cd pretrained
cp /path/to/gta5_only.pth ./gta5_only.pth
cp /path/to/synthia_only.pth ./synthia_only.pth
```

- generate pseudo labels

```bash
# pseudo labels will be saved in ./experiments/gta5_pseudo_label/
# for detailed arguments, please refer to the codes
python generate_pseudo_labels.py
```

- train the baseline model (i.e., the cross-domain distillation stage)

```bash
python main.py \
--module-name baseline \
--mode train \
--restore-from ./pretrained/gta5_only.pth  \
--experiment-name none \
--train-batch-size 4 \
--lr 1.0e-4 \
--val-interval 1.0 \
--pseudo-label-root ./experiments/gta5_pseudo_label/
```

- extract the baseline model state_dict from lightning checkpoint file

```bash
python extract_model_state_dict.py \
--pl-weight-path /path/to/your/trained-module-state-file.ckpt \
--model-weight-keyname net \
--save-path ./pretrained/basic.pth
```

- mine hard patches from the source-pre-trained model

```bash
python construct_hard_patch.py \
--/path/to/cityscapes \
--data-dir /path/to/cityscapes \
--data-list ./datasets/list/cityscapes/train.txt \
--restore-from ./pretrained/gta5_only.pth \
--save ./experiments/hard_patch_from_gta5_model/
```

- train the target model with the proposed spatial_mix and hard patches mining mechanisms (e.g., self-distillation)

```bash
python main.py \
--module-name smhm \
--mode train \
--restore-from ./pretrained/basic.pth  \
--experiment-name none \
--train-batch-size 4 \
--lr 1.0e-5 \
--val-interval 1.0 \
--position-beta 0.25 \
--hard-patches-root ./experiments/hard_patch_from_gta5_model/
```

- repeat several times of self-ditillation for the optimal model

## Acknowledge

We thank the authors of AdaptSegNet[https://github.com/wasidennis/AdaptSegNet] for their SourceOnly model.

Part of our implementation also refers to ClassMix[https://github.com/WilhelmT/ClassMix] and CutMix[https://github.com/clovaai/CutMix-PyTorch],
and we extend our gratitude to the authors.
