# CROTS: CROss-domain Teacher-Student Learning for Source-Free Domain Adaptive Semantic Segmentation

## Abstract

```tex
Source-free domain adaptation (SFDA) aims to transfer source knowledge to target domain from pre-trained source models without accessing to private source data. Existing SFDA methods typically adopt the self-training strategy employing the pre-trained source model to generate pseudo-labels for unlabeled target data. However, these methods are subject to strict limitations: 1) The discrepancy between source and target domains results in intense noise and unreliable pseudo-labels. Overfitting noisy pseudo-labeled target data will lead to drastic performance degradation. 2) Considering the class-imbalanced pseudo-labels, the target model is prone to forget the minority classes. Aiming at these two limitations, this study proposes a \textbf{\underline{CRO}}ss domain \textbf{\underline{T}}eacher-\textbf{\underline{S}}tudent learning framework (namely \textbf{\underline{CROTS}}) to achieve source-free domain adaptive semantic segmentation.  
Specifically, with pseudo-labels provided by the intra-domain teacher model, CROTS incorporates Spatial-Aware Data Mixing to generate diverse samples by randomly mixing different patches respecting to their spatial semantic layouts, which boosts the diversity of training data and avoids the overfitting issue.
Meanwhile, during inter-domain teacher-student learning, CROTS fosters Hard Patches Mining strategy to mitigate the class imbalance phenomenon. To this end, the inter-domain teacher model helps exploit samples of long-tailed rare classes and increase their contribution to student learning, which regularizes the student model to avoid forgetting them. Extensive experimental results have demonstrated that: 1) CROTS mitigates the overfitting issue and contributes to stable performance improvement, i.e., +16.0\% mIoU and +16.5\% mIoU for SFDA in GTA5$\to$Cityscapes and SYNTHIA$\to$Cityscapes, respectively; 2) CROTS improves task performance for long-tailed rare classes, alleviating the issue of class imbalance; 3) CROTS achieves superior performance to other leading SFDA counterparts (54.2\% mIoU and 60.3\% mIoU for the above two SFDA benchmarks, respectively); 4) CROTS can be applied under the black-box SFDA setting (53.7\% mIoU and 59.3\% mIoU for the above two SFDA benchmarks, respectively), even outperforming many white-box SFDA methods.
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
