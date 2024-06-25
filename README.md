# Consistent Image Synthesis for T2I Personalization 

**📢2024년 여름 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.**

**🎉2024년 여름 AIKU Conference 열심히상 수상!**

## 소개



## 참고 논문

> **Custom Diffusion** [[repo]](https://github.com/CompVis/latent-diffusion)
>
> _Proposed in [“High-Resolution Image Synthesis with Latent Diffusion Models”](https://arxiv.org/abs/2112.10752),
> CVPR 2022

> **RePaint** [[repo]](https://github.com/andreas128/RePaint)
>
> _Proposed in [“RePaint: Inpainting using Denoising Diffusion Probabilistic Models”](https://arxiv.org/abs/2201.09865),
> CVPR 2022

> **ControlNet** [[repo]](https://github.com/lllyasviel/ControlNet)
>
> _Proposed in [“Adding Conditional Control to Text-to-Image Diffusion Models"](https://arxiv.org/abs/2302.05543),
> CVPR 2023

## 방법론 1: Cross Attention Map 

Stable Diffusion 모델을 활용하여 text condition을 주고, 1차적으로 잘린 이미지를 context에 맞게 생성합니다. 그러나 text prompt만으로는 Stable Diffusion이 복잡한 pose에 대한 semantic를 정확히 이해하기 어렵습니다. 이 문제를 해결하기 위해, Openpose를 사용하여 이미지에서 pose를 추출하고 이를 수정합니다. 이후, contextualize가 충분히 이루어지지 않은 부분에 대한 mask와 함께 ControlNet의 condition으로 제공하여 inpainting 작업을 수행합니다.

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



## 팀원

- [김민재](https://github.com/mingming2000): RePaint implementation (Stable Diffusion), Outpainting & ControlNet
- [지동환](https://github.com/kmjnwn): RePaint implementation (Stable Diffusion, DDIM)
- [김민영](https://github.com/kwjames98): Paper research, Demo
- [황정현](): 
