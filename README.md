# Consistent Image Synthesis for T2I Personalization 

**📢2024년 여름 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.**

**🎉2024년 여름 AIKU Conference 열심히상 수상!**

## 소개

본 프로젝트는 
T2I personalization task는 사용자 제공 reference image를 기반으로 T2I diffusion 모델을 사용자 맞춤화하는 것입니다. 몇 장의 reference image와 text prompt를 제공하면 다양한 pose, view, background에서 대상의 새로운 렌더링을 생성할 수 있습니다. 기존 접근 방식은 고유한 텍스트 임베딩을 사용하여 대상을 나타내며, 텍스트 임베딩 자체나 확산 모델의 매개변수를 최적화하여 대상을 표현합니다. 그러나 이러한 방법들은 종종 색상, 텍스처 및 모양과 같은 대상의 외관을 정확하게 모방하는 데 실패합니다. 이는 텍스트 임베딩이 대상의 시각적 외관을 표현하는 데 충분한 spatial representation을 가지지 못하기 때문입니다.

## 팀원

| 팀원                            | 역할                                       |
| -------------------------------------- | ---------------------------------------- |
| [김민재](https://github.com/kwjames98)*      | Inference, Code analysis(Textual Inversion),  Paper(Abstract, Introduction, Related Works, Preliminary, Method)  |
| [김민영](https://github.com/EuroMinyoung186)     | Code analysis(Cross attention map), Inference, Evaluation(MasaCtrl), Distributive processing, Paper(Experiments, Conclusion) |
| [지동환](https://github.com/zheedong)                          | Code analysis(Textual Inversion, Cross attention map), Inference, Paper(Reference) |
| [황정현](https://github.com/imjunghyunee)                           | Code analysis(Textual Inversion), Paper(Experiments, Conclusion), Inference |

## 참고 논문

> **Custom Diffusion** [[repo]](https://github.com/adobe-research/custom-diffusion)
>
> _Proposed in [“Multi-Concept Customization of Text-to-Image Diffusion”](https://arxiv.org/abs/2212.04488),
> CVPR 2023

> **MasaCtrl** [[repo]](https://github.com/TencentARC/MasaCtrl)
>
> _Proposed in [“MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing”](https://arxiv.org/abs/2304.08465),
> ICCV 2023


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
- [황정현](https://github.com/imjunghyunee)
