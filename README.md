# Unsupervised Domain Adaptation via Style-Aware Self-intermediate Domain

Code release for "Unsupervised Domain Adaptation via Style-Aware Self-intermediate Domain" (Pattern Recognization)

## Paper

<div align=center><img src="https://github.com/LyWang12/SSID/blob/main/Fig/fig_ssid.pdf" width="100%"></div>

[Unsupervised Domain Adaptation via Style-Aware Self-intermediate Domain](https://www.sciencedirect.com/science/article/pii/S0031320325010052) 
(Pattern Recognization)

We propose the Style-aware Self-Intermediate Domain (SSID), a novel framework that bridges large domain gaps by mimicking human transitive inference, using a sequence of style-aware synthesized intermediate domains to preserve discriminative features during knowledge transfer.



## Datasets

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### VisDA 2017

VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public).

<div align=center><img src="https://github.com/LyWang12/SSID/blob/main/Fig/fig_exp.pdf" width="100%"></div>


## Running the code

```
python train_SWIN_T+I+loss_home.py
```

## Citation
If you find this code useful for your research, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Model_Barrier_A_Compact_Un-Transferable_Isolation_Domain_for_Model_Intellectual_CVPR_2023_paper.html):
```
@article{wang2025unsupervised,
  title={Unsupervised domain adaptation via style-aware self-intermediate domain},
  author={Wang, Lianyu and Wang, Meng and Zhang, Daoqiang and Fu, Huazhu},
  journal={Pattern Recognition},
  pages={112344},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements
Some codes are adapted from [NTL](https://github.com/conditionWang/NTL) and 
[SWIN-Transformer](https://github.com/microsoft/Swin-Transformer). We thank them for their excellent projects.

## Contact
If you have any problem about our code, feel free to contact
- lywang12@126.com
- wangmeng9218@126.com

or describe your problem in Issues.
