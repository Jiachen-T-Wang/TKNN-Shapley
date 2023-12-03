# Threshold KNN-Shapley: A Linear-Time and Privacy-Friendly Approach to Data Valuation

Jiachen T. Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, Prateek Mittal
> 
> Data valuation aims to quantify the usefulness of individual data sources in training machine learning (ML) models, and is a critical aspect of data-centric ML research. However, data valuation faces significant yet frequently overlooked privacy challenges despite its importance. This paper studies these challenges with a focus on KNN-Shapley, one of the most practical data valuation methods nowadays. We first emphasize the inherent privacy risks of KNN-Shapley, and demonstrate the significant technical difficulties in adapting KNN-Shapley to accommodate differential privacy (DP). To overcome these challenges, we introduce TKNN-Shapley, a refined variant of KNN-Shapley that is privacy-friendly, allowing for straightforward modifications to incorporate DP guarantee (DP-TKNN-Shapley). We show that DP-TKNN-Shapley has several advantages and offers a superior privacy-utility tradeoff compared to naively privatized KNN-Shapley in discerning data quality. Moreover, even non-private TKNN-Shapley achieves comparable performance as KNN-Shapley. Overall, our findings suggest that TKNN-Shapley is a promising alternative to KNN-Shapley, particularly for real-world applications involving sensitive data.

<a href="https://arxiv.org/abs/2308.15709"><img src="https://img.shields.io/badge/arXiv-2308.15709-b31b1b.svg" height=22.5></a>

<p align="center">
<img src="assets/scenario.png" width="600px"/>  
<br>
</p>

## Description

Official implementation of our NeurIPS 2023 Spotlight paper [Threshold KNN-Shapley: A Linear-Time and Privacy-Friendly Approach to Data Valuation](https://arxiv.org/pdf/2308.15709.pdf), where we propose a new data valuation technique that is **training-free**, has **linear runtime**, and can be **easily modified to provide provable privacy guarantee**. 

If you have any questions related to the code or the paper, feel free to email **Jiachen** (tianhaowang@princeton.edu) and **Ruoxi** (ruoxijia@vt.edu). 

## Quick Start
To reproduce the mislabeled data detection experiments in our paper, we can simply run the following:
```
python main.py --task mislabel_detect --dataset 2dplanes --value_type TNN-SV --n_data 2000 --n_val 200 --flip_ratio 0.1 --tau -0.5 --n_repeat 1
```
This command will give you the AUROC score of milabeled detection task of TKNN-Shapley on 2dplanes dataset:
```
-------
Load Dataset 2dplanes
# of classes = 2
-------
Data Error Type: Mislabel
-------
tau in tnn shapley -0.5
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 887.83it/s]
Data Value Computed; Value Name: TNN-SV; Runtime: 0.228 s
Task: mislabel_detect
*** TNN-SV AUROC: 0.92 (0.0), eps=inf, delta=0***
```


Here's the naming of `value_type`:
- KNN-SV-RJ: original KNN-Shapley [1].
- KNN-SV-JW: soft-label KNN-Shapley [2]. 
- KNN-SV-RJ-private: naively privatized KNN-Shapley (without subsampling), described in our Appendix B.4. 
- KNN-SV-RJ-private-withsub: naively privatized KNN-Shapley (with subsampling), described in our Appendix B.4. 
- TNN-SV: Threshold KNN-Shapley, described in our Section 4. 
- TNN-SV-private: Private version of Threshold KNN-Shapley, described in our Section 5. 


For private setting, we need to also specify the desired privacy parameter `eps`, `delta`, and subsampling rate `q`; moreover, we can run the experiment for multiple times to obtain the variance of the results. 
The following command line gives you the result for DP-TKNN-Shapley:
```
python main.py --task mislabel_detect --dataset 2dplanes --value_type TNN-SV-private --n_data 2000 --n_val 200 --flip_ratio 0.1 --tau -0.5 --eps 0.1 --delta 1e-4 --n_repeat 5 --q 0.01
```
where the output is 
```
-------
Load Dataset 2dplanes
# of classes = 2
-------
Data Error Type: Mislabel
-------
Noise magnitude sigma=3.7952708147564307
Noise magnitude sigma=3.7952708147564307
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 687.82it/s]
Data Value Computed; Value Name: TNN-SV-private; Runtime: 1.889 s
Noise magnitude sigma=3.7952708147564307
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 685.21it/s]
Data Value Computed; Value Name: TNN-SV-private; Runtime: 0.293 s
Noise magnitude sigma=3.7952708147564307
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 702.68it/s]
Data Value Computed; Value Name: TNN-SV-private; Runtime: 0.286 s
Noise magnitude sigma=3.7952708147564307
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 724.12it/s]
Data Value Computed; Value Name: TNN-SV-private; Runtime: 0.277 s
Noise magnitude sigma=3.7952708147564307
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 712.55it/s]
Data Value Computed; Value Name: TNN-SV-private; Runtime: 0.281 s
Task: mislabel_detect
*** TNN-SV-private AUROC: 0.885 (0.012), eps=0.1, delta=0.0001***
```



## Citation

If you use this code in your research, please cite the following work:
```bibtex
@article{wang2023threshold,
  title={Threshold KNN-Shapley: A Linear-Time and Privacy-Friendly Approach to Data Valuation},
  author={Wang, Jiachen T and Zhu, Yuqing and Wang, Yu-Xiang and Jia, Ruoxi and Mittal, Prateek},
  journal={arXiv preprint arXiv:2308.15709},
  year={2023}
}
```

## Reference
[1] Jia, Ruoxi, et al. "Efficient task-specific data valuation for nearest neighbor algorithms." VLDB 2019

[2] Wang, Jiachen T., and Ruoxi Jia. "A Note on" Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms"." Technical Note (2023).
