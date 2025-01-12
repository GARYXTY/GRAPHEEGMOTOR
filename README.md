# GRAPHEEGMOTOR
Source Code for Learning EEG Motor Characteristics via Temporal-Spatial Representations
This paper is accepted by IEEE Transactions on Emerging Topics in Computational Intelligence.

### Overview

This repository provides the main code for the paper, along with training and testing notebooks to facilitate quick experimentation on different datasets. We also provide the trained weights and preprocessed data (Creat a folder named "data" and download data through this link: https://pan.quark.cn/s/a5d9239401bb) for the Physionet dataset using a subject-dependent paradigm.

The implementation references the repository
"https://github.com/vermaMachineLearning/keras-deep-graph-learning"

If you find this repository helpful, please cite our paper:
Tianyu Xiang, Xiaohu Zhou, Xiaoliang Xie, Shiqi Liu, Hongjun Yang, Fengzhen Qiu, Meijiang Gui, Hao Li, DeXing Huang, XiuLing Liu, and Zengguang Hou, [“Learning EEG motor characteristics via temporal-spatial representations”](https://ieeexplore.ieee.org/abstract/document/10663067), IEEE Transactions on Emerging Topics in Computational Intelligence, 2024.

###  IMPORTANT NOTICE

We sincerely appreciate the insightful observation by **Xiangguo Yin** from Shandong University, which brought to our attention discrepancies between the implementation in our code and the method described in our paper.

![revised_paper](https://github.com/user-attachments/assets/87d9edba-38d9-4e8e-a9e9-c111a10fa58b)

As illustrated in the figure, the input to the temporal block in the implementation consists solely of the temporal representation. After passing through two layers of 1D CNN, the temporal representation is input into the Res-block. The output from the Res-block is added to either the original temporal representation or the temporal representation processed by CNN and max pooling to produce the downsampled temporal representation. The input to the Res-block was also adjusted accordingly. Other components remain consistent with the description in the paper.

**All results reported in our paper were obtained using the implementation provided in this repository.** While there are some differences in the implementation details, the overall methodology and motivation of the paper remain consistent. After being notified of the discrepancy, we attempted to modify the implementation to align with the method described in the paper. However, this resulted in a decline in experimental performance, particularly in the temporal stream, likely due to gradient vanishing.

For those interested in continuing work in this area, we recommend referring to the implementation provided in this repository.

------

We welcome discussions and collaborations. Feel free to contact us at xiangtianyu2021@ia.ac.cn.
