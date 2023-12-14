<p align="center" width="100%">
<img src="assets\SeisCLIP.png"  width="80%" height="80%">
</p>


<div>
<div align="center">
    <a href='https://sixu0.github.io/' target='_blank'>Xu Si<sup>1</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Xinming  Wu<sup>1,†,‡</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Hanlin Sheng<sup>1</sup></a>&emsp;
    </br>
    <a href='https://dams.ustc.edu.cn/main.htm' 
    target='_blank'>Jun Zhu<sup>1</sup></a>&emsp;
    <a href='https://dams.ustc.edu.cn/main.htm' 
    target='_blank'>Zefeng Li<sup>1</sup></a>&emsp;
</div>
<div>

<div align="center">
    <sup>1</sup>
    University of Science and Technology of China&emsp;
    </br>
    <!-- <sup>*</sup> Equal Contribution&emsp; -->
    <sup>†</sup> Corresponding Author&emsp;
    <sup>‡</sup> Project Lead&emsp;
</div>

-----------------

[![arXiv](https://img.shields.io/badge/arxiv-2309.02320-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2309.02320)](https://arxiv.org/abs/2309.02320)

### 🌟 Spec-based Foundation Model Supports A Wide Range of Seismology


 As shown in this figure, SeisCLIP can provide services for downstream tasks including event classification 💥 , location 🌍 , mechanism ⛰, etc.


# 🌟 News

* **2023.9.14:** 🌟🌟🌟 Pretrained weight and a simple usage demo for out SeisCLIP have been released. The implementation of SeisCLIP for event classification also released. Because the location and focal mechanism analysis code need lib 'Pytorch_geometric', it may be challenging for beginners. To provide a more detailed documentation, we will release it later. (Python Version 3.9.0 is recommended)
* **2023.9.8:** Paper is released at [arxiv](https://arxiv.org/abs/2309.02320), and code will be gradually released.
* **2023.8.7:** Github Repository Initialization. (copy README template from Meta-Transformer)


# 🔓 Model Zoo

<!-- <details> -->
<summary> Open-source Modality-Agnostic Models </summary>
<br>
<div>

|      Model      |   Pretraining   | Spec Size | #Param |                                               Download | 国内下载源                                               |
| :------------: | :----------: | :----------------------: | :----: | :---------------------------------------------------------------------------------------------------: | :--------: | 
| SeisCLIP  | STEAD-1M |         50 × 120          |  -  |   [ckpt](https://drive.google.com/file/d/1UIeFWl2wENr83GRtdi6Tlj4MLDZ3UctN/view?usp=drive_link)     | [ckpt]
| SeisCLIP  | STEAD-1M |         50 × 600          |  -  |   [ckpt](https://drive.google.com/file/d/1_YiqeaBlBg-EKJ36Yvluoc50n4Y86aI3/view?usp=drive_link)     | [ckpt]


&ensp;
# Citation
If the code and paper help your research, please kindly cite:
```
@article{si2023seisclip,
        title={SeisCLIP: A seismology foundation model pre-trained by multi-modal data for multi-purpose seismic feature extraction}, 
        author={Xu Si and Xinming Wu and Hanlin Sheng and Jun Zhu and Zefeng Li},
        year={2023},
        journal={arXiv preprint arXiv:2309.02320},
}
```
# License
This project is released under the [MIT license](LICENSE).

# Acknowledgement
This code is developed based on excellent open-sourced projects including [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main), [AST](https://github.com/YuanGongND/ast), [MetaTransformer](https://github.com/invictus717/MetaTransformer/tree/master), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [Seisbench](https://github.com/seisbench/seisbench), [STEAD](https://github.com/smousavi05/STEAD) and [PNW](https://github.com/niyiyu/PNW-ML).
