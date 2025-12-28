# NeurIPS 2025 papers (OpenReview export)

- **Generated (UTC)**: 2025-12-25 16:46:07Z
- **Count**: 5

## Index

- [ModHiFi: Identifying High Fidelity predictive components for Model Modification](#modhifi-identifying-high-fidelity-predictive-components-for-model-modification)
- [REVE: A Foundation Model for EEG - Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects](#reve-a-foundation-model-for-eeg-adapting-to-any-setup-with-large-scale-pretraining-on-25000-subjects)
- [The Structure of Relation Decoding Linear Operators in Large Language Models](#the-structure-of-relation-decoding-linear-operators-in-large-language-models)
- [Time-o1: Time-Series Forecasting Needs Transformed Label Alignment](#time-o1-time-series-forecasting-needs-transformed-label-alignment)
- [Vulnerable Data-Aware Adversarial Training](#vulnerable-data-aware-adversarial-training)

## ModHiFi: Identifying High Fidelity predictive components for Model Modification

- **Authors**: Dhruva Kashyap, Chaitanya Murti, Pranav K Nayak, Tanay Narshana, Chiranjib Bhattacharyya
- **Venue**: NeurIPS 2025 spotlight
- **OpenReview**: `https://openreview.net/forum?id=lClK4uBxSG`
- **PDF**: `https://openreview.net/pdf/e8b0a7733f128946bbbf4c9b65ec1c7014eb64b0.pdf`

### Abstract

Modifying well-trained models for purposes such as pruning or unlearning, without access to training data or the original loss function, is a challenging problem. While techniques exist for such modification, they often require training data, are computationally expensive, or are architecture-specific. To address this, we investigate the fundamental question of identifying components that are critical to the model’s predictive performance, without access to either gradients or the loss function, and with only distributional access such as synthetic data. 
    We theoretically demonstrate that the global reconstruction error is linearly bounded by local reconstruction errors for Lipschitz-continuous networks such as CNNs and well-trained Transformers (which, contrary to existing literature, we find exhibit Lipschitz continuity). This motivates using the locally reconstructive behavior of component subsets to quantify their global importance, via a metric that we term *Subset Fidelity*. In the uncorrelated features setting, selecting individual components via their Subset Fidelity scores is optimal, which we use to propose **ModHiFi**, an algorithm for model modification that requires no training data or loss function access. **ModHiFi-P**, for structured pruning, achieves an 11% speedup over the current state of the art on ImageNet models and competitive performance on language models. **ModHiFi-U**, for classwise unlearning, achieves complete unlearning on CIFAR-10 without fine-tuning and demonstrates competitive performance on Swin Transformers.

### BibTeX

```
@inproceedings{
kashyap2025modhifi,
title={ModHiFi: Identifying High Fidelity predictive components for Model Modification},
author={Dhruva Kashyap and Chaitanya Murti and Pranav K Nayak and Tanay Narshana and Chiranjib Bhattacharyya},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=lClK4uBxSG}
}
```

## REVE: A Foundation Model for EEG - Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects

- **Authors**: Yassine El Ouahidi, Jonathan Lys, Philipp Thölke, Nicolas Farrugia, Bastien Pasdeloup, Vincent Gripon, Karim Jerbi, Giulia Lioi
- **Venue**: NeurIPS 2025 poster
- **OpenReview**: `https://openreview.net/forum?id=ZeFMtRBy4Z`
- **PDF**: `https://openreview.net/pdf/0b9fb85ecc9a62002866ac3fb9367e22d13fa0db.pdf`

### Abstract

Foundation models have transformed AI by reducing reliance on task-specific data through large-scale pretraining. While successful in language and vision, their adoption in EEG has lagged due to the heterogeneity of public datasets, which are collected under varying protocols, devices, and electrode configurations. Existing EEG foundation models struggle to generalize across these variations, often restricting pretraining to a single setup, resulting in suboptimal performance, in particular under linear probing.
We present REVE (Representation for EEG with Versatile Embeddings), a pretrained model explicitly designed to generalize across diverse EEG signals. REVE introduces a novel 4D positional encoding scheme that enables it to process signals of arbitrary length and electrode arrangement. Using a masked autoencoding objective, we pretrain REVE on over 60,000 hours of EEG data from 92 datasets spanning 25,000 subjects, representing the largest EEG pretraining effort to date.
REVE achieves state-of-the-art results on 10 downstream EEG tasks, including motor imagery classification, seizure detection, sleep staging, cognitive load estimation, and emotion recognition. With little to no fine-tuning, it demonstrates strong generalization, and nuanced spatio-temporal modeling. We release code, pretrained weights, and tutorials to support standardized EEG research and accelerate progress in clinical neuroscience.

### BibTeX

```
@inproceedings{
ouahidi2025reve,
title={{REVE}: A Foundation Model for {EEG} - Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects},
author={Yassine El Ouahidi and Jonathan Lys and Philipp Th{\"o}lke and Nicolas Farrugia and Bastien Pasdeloup and Vincent Gripon and Karim Jerbi and Giulia Lioi},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=ZeFMtRBy4Z}
}
```

## The Structure of Relation Decoding Linear Operators in Large Language Models

- **Authors**: Miranda Anna Christ, Adrián Csiszárik, Gergely Becsó, Dániel Varga
- **Venue**: NeurIPS 2025 spotlight
- **OpenReview**: `https://openreview.net/forum?id=XsBzmJzJ2l`
- **PDF**: `https://openreview.net/pdf/ed3dd009ac0424ecb2d27ab9be7041714b6d8359.pdf`

### Abstract

This paper investigates the structure of linear operators introduced in Hernandez et al. [2023] that decode specific relational facts in transformer language models. We extend their single-relation findings to a collection of relations and systematically chart their organization. We show that such collections of relation decoders can be highly compressed by simple order-3 tensor networks without significant loss in decoding accuracy. To explain this surprising redundancy, we develop a cross-evaluation protocol, in which we apply each linear decoder operator to the subjects of every other relation. Our results reveal that these linear maps do not encode distinct relations, but extract recurring, coarse-grained semantic properties (e.g., country of capital city and country of food are both in the country-of-X property). This property-centric structure clarifies both the operators' compressibility and highlights why they generalize only to new relations that are semantically close. Our findings thus interpret linear relational decoding in transformer language models as primarily property-based, rather than relation-specific.

### BibTeX

```
@inproceedings{
christ2025the,
title={The Structure of Relation Decoding Linear Operators in Large Language Models},
author={Miranda Anna Christ and Adri{\'a}n Csisz{\'a}rik and Gergely Becs{\'o} and D{\'a}niel Varga},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=XsBzmJzJ2l}
}
```

## Time-o1: Time-Series Forecasting Needs Transformed Label Alignment

- **Authors**: Hao Wang, Licheng Pan, Zhichao Chen, Xu Chen, Qingyang Dai, Lei Wang, Haoxuan Li, Zhouchen Lin
- **Venue**: NeurIPS 2025 poster
- **OpenReview**: `https://openreview.net/forum?id=RxWILaXuhb`
- **PDF**: `https://openreview.net/pdf/a2b6688d0676976900536577d2fada37d0e39c76.pdf`

### Abstract

Training time-series forecast models presents unique challenges in designing effective learning objectives. Existing methods predominantly utilize the temporal mean squared error, which faces two critical challenges: (1) label autocorrelation, which leads to bias from the label sequence likelihood; (2) excessive amount of tasks, which increases with the forecast horizon and complicates optimization. To address these challenges, we propose Time-o1, a transformation-augmented learning objective for training time-series forecasting models. The central idea is to transform the label sequence into decorrelated components with discriminated significance. Models are then trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Extensive experiments demonstrate that Time-o1 achieves state-of-the-art performance and is compatible with various forecast models. Code is available at https://github.com/Master-PLC/Time-o1.

### BibTeX

```
@inproceedings{
wang2025timeo,
title={Time-o1: Time-Series Forecasting Needs Transformed Label Alignment},
author={Hao Wang and Licheng Pan and Zhichao Chen and Xu Chen and Qingyang Dai and Lei Wang and Haoxuan Li and Zhouchen Lin},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=RxWILaXuhb}
}
```

## Vulnerable Data-Aware Adversarial Training

- **Authors**: Yuqi Feng, Jiahao Fan, Yanan Sun
- **Venue**: NeurIPS 2025 poster
- **OpenReview**: `https://openreview.net/forum?id=yrrU5YChQr`
- **PDF**: `https://openreview.net/pdf/0c873f68a38655feeef37be13f358ee0fb16d4da.pdf`

### Abstract

Fast adversarial training (FAT) has been considered as one of the most effective alternatives to the computationally-intensive adversarial training. Generally, FAT methods pay equal attention to each sample of the target task. However, the distance between each sample and the decision boundary is different, learning samples which are far from the decision boundary (i.e., less important to adversarial robustness) brings additional training cost and leads to sub-optimal results. To tackle this issue, we present vulnerable data-aware adversarial training (VDAT) in this study. Specifically, we first propose a margin-based vulnerability calculation method to measure the vulnerability of data samples. Moreover, we propose a vulnerability-aware data filtering method to reduce the training data for adversarial training thus improve the training efficiency. The experiments are conducted in terms of adversarial training and robust neural architecture search on CIFAR-10, CIFAR-100, and ImageNet-1K. The results demonstrate that VDAT is up to 76% more efficient than state-of-the-art FAT methods, while achieving improvements regarding the natural accuracy and adversarial accuracy in both scenarios. Furthermore, the visualizations and ablation studies show the effectiveness of both core components designed in VDAT.

### BibTeX

```
@inproceedings{
feng2025vulnerable,
title={Vulnerable Data-Aware Adversarial Training},
author={Yuqi Feng and Jiahao Fan and Yanan Sun},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=yrrU5YChQr}
}
```
