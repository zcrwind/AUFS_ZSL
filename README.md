# AUFS_ZSL
The source code of our IJCAI 2018 conference paper ["Visual Data Synthesis via GAN for Zero-shot Video Classification"](https://www.ijcai.org/Proceedings/2018/157), where "AUFS" is short for "Adversarial Unseen Feature Synthesis".

## Introduction
Zero-Shot Learning (ZSL) in video classification is a promising research direction, which aims to tackle the challenge from explosive growth of video categories. Most existing methods exploit seento- unseen correlation via learning a projection between visual and semantic spaces. However, such projection-based paradigms cannot fully utilize the discriminative information implied in data distribution, and commonly suffer from the information degradation issue caused by “heterogeneity gap”. In this paper, we propose a visual data synthesis framework via GAN to address these problems. Specifically, both semantic knowledge and visual distribution are leveraged to synthesize video feature of unseen categories, and ZSL can be turned into typical supervised problem with the synthetic features. First, we propose multi-level semantic inference to boost video feature synthesis, which captures the discriminative information implied in joint visual-semantic distribution via feature-level and label-level semantic inference. Second, we propose Matching-aware Mutual Information Correlation to overcome information degradation issue, which captures seen-to-unseen correlation in matched and mismatched visual-semantic pairs by mutual information, providing the zero-shot synthesis procedure with robust guidance signals. Experimental results on four video datasets demonstrate that our approach can improve the zero-shot video classification performance significantly.

## Dependencies
- python 3.7+
- pytorch 1.0.0+
- torchvision 0.2.1+
- numpy 1.15.4+
- scipy 1.2.0+
- scikit-learn 0.21.2+
- CUDA 10.0+
- cudnn 6.0.21+


## Datasets
It is noted that though our original paper aims to tackle zero-shot learning problem in video domains, our approach is feature-agnostic, thus it is straightforward to extend our proposal to ZSL in image domains. I cannot access the video datasets used in the original paper because of the change of mentor, so I report the performance of our AUFS on the widely-used image datasets for ZSL. As shown in the table below, our proposal achieves the ZSL performance comparable to the state-of-the-art methods as of 2018.
- **aPY**. Attribute Pascal and Yahoo (aPY) is a small-scale coarse-grained dataset with 64 attributes.
- **CUB** Caltech-UCSDBirds 200-2011 (CUB) is a fine-grained and medium scale dataset with respect to both number of images and number of classes, i.e. 11, 788 images from 200 different types of birds annotated with 312 attributes.
- **AwA1** Animals with Attributes (AWA1) is a coarse-grained dataset that is medium-scale in terms of the number of images, i.e. 30, 475 and small-scale in terms of number of classes, i.e. 50 classes.
- **AwA2** Animals with Attributes2 (AWA2) is introduced by [9], which contains 37, 322 images for the 50 classes of AWA1 dataset from public web sources, i.e. Flickr, Wikipedia, etc., making sure that all images of AWA2 have free-use and redistribution licenses and they do not overlap with images of the original Animal with Attributes dataset.
- **SUN** SUN is a fine-grained and medium-scale dataset with respect to both number of images and number of classes, i.e. SUN contains 14340 images coming from 717 types of scenes annotated with 102 attributes.

## Experimental Results
|Dataset|aPY|CUB|AwA1|AwA2|SUN|
|--|--|--|--|--|--|
|DAP [1]|33.8|40.0|44.1|46.1|39.9|
|IAP [1]|36.6|24.0|35.9|35.9|19.4|
|CONSE [2]|26.9|34.3|45.6|44.5|38.8|
|CMT [3]|28.0| 34.6|39.5 |37.9|39.9|
|SSE [4]|34.0|43.9|60.1|61.0| 51.5|
|LATEM [5]|35.2|49.3|55.1|55.8|55.3|
|DEVISE [6]|39.8| 52.0|54.2| 59.7|56.5|
|SJE [7]|32.9|53.9|65.6|61.9| 53.7|
|ESZSL [8]|38.3|53.9|58.2| 58.6|54.5|
|Ours|39.1|54.2|59.7|58.5|55.3|

## References
[1] C. Lampert, H. Nickisch, and S. Harmeling, “Attribute-based classification for zero-shot visual object categorization,” in TPAMI, 2013.  
[2] M. Norouzi, T. Mikolov, S. Bengio, Y. Singer, J. Shlens, A. Frome, G. Corrado, and J. Dean, “Zero-shot learning by convex combination
of semantic embeddings,” in ICLR, 2014.  
[3] R. Socher, M. Ganjoo, C. D. Manning, and A. Ng, “Zero-shot learning through cross-modal transfer,” in NIPS, 2013.  
[4] Z. Zhang and V. Saligrama, “Zero-shot learning via semantic similarity embedding,” in ICCV, 2015.  
[5] Y. Xian, Z. Akata, G. Sharma, Q. Nguyen, M. Hein, and B. Schiele, “Latent embeddings for zero-shot classification,” in CVPR, 2016.  
[6] A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean, M. A. Ranzato, and T. Mikolov, “Devise: A deep visual-semantic embedding model,” in NIPS, 2013, pp. 2121–2129.  
[7] Z. Akata, S. Reed, D. Walter, H. Lee, and B. Schiele, “Evaluation of output embeddings for fine-grained image classification”, in CVPR, 2015.  
[8] B. Romera-Paredes and P. H. Torr, “An embarrassingly simple approach to zero-shot learning,” ICML, 2015.  
[9] Yongqin Xian, Bernt Schiele1 and Zeynep Akata, "Zero-Shot Learning - The Good, the Bad and the Ugly", in CVPR 2018.

If you make use of this code or the adversarial unseen feature synthesis algorithm in your work, please cite the paper:
```
@inproceedings{zhang2018Visual,
	title={Visual Data Synthesis via GAN for Zero-Shot Video Classification},
	author={Zhang, Chenrui and Peng, Yuxin},
	booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence},
	pages={1128--1134},
	year={2018}
}
```


flag