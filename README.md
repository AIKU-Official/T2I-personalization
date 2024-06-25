# Consistent Image Synthesis for T2I Personalization 

**📢2024년 여름 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.**

**🎉2024년 여름 AIKU Conference 열심히상 수상!**

## 소개



## 참고 논문

> **Custom Diffusion** [[repo]](https://github.com/adobe-research/custom-diffusion)
>
> _Proposed in [“Multi-Concept Customization of Text-to-Image Diffusion”](https://arxiv.org/abs/2212.04488),
> CVPR 2022

> **MasaCtrl** [[repo]](https://github.com/TencentARC/MasaCtrl)
>
> _Proposed in [“MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing”](https://arxiv.org/abs/2304.08465),
> CVPR 2022


## 방법론 1: Cross Attention Map 



### MasaCtrl Architecture
<p align="center">
    <img src = "./image/architecture.png"
        style = "width: 70%">
</p>



<p align="center">
    <img src = "./image/only_realisticvision_output.png"
        style = "width: 50%">
</p>

<p align="center">
    <img src = "./image/outpainting_output.png"
        style = "width: 40%">
</p>


<p align="center">
    <img src = "./image/openpose_editor.gif"
        style = "width: 70%">
</p>

## 방법론 2: Denoising Step and layer


<p align="center">
    <img src = "./image/repaint_architecture.png"
        style = "width: 70%">
</p>

### 1. Denoising step



### 2. Layer


## 예시 결과

### final step별 image 차이
<img src = "./images/corgi_final_step.png">

#### Encoder와 Decoder별 image 차이
<img src = "./images/corgi_layer_type.png">

#### step 간격 별 image 차이
<img src = "./images/corgi_skip_step_interval.png">

#### start step별 image 차이
<img src = "./images/corgi_start_step.png">

#### final_step별 text-alignment와 image-alignment 사이의 관계
<img src = "./images/final_step_plot.png">

#### skip_step별 text-alignment와 image-alignment 사이의 관계 
<img src = "./images/skip_step_interval_plot.png">

#### start_step별 text-alignment와 image-alignment 사이의 관계 
<img src = "./images/start_step_plot.png">

## 팀원

- [김민영](https://github.com/EuroMinyoung186)
- [지동환](https://github.com/zheedong)
- [김민재](https://github.com/kwjames98)
- [황정현]()
