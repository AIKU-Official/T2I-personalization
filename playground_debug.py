import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

from masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMaskAuto
from torchvision.io import read_image

from masactrl.utils import get_total_step_layer
from transformers import CLIPProcessor, CLIPModel

from masactrl.evaluate_clip import calculate_clip_s_i2t, calculate_clip_s

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def load_dataset(path, selected_object = None):
    with open(path, 'r') as f:
        objects = json.load(f)

    if selected_object is None:
        return_lst = []
        for object in objects:
            return_lst.append((object['name'], object['description'], object['image_path']))
        return return_lst
    else:
        for object in objects:
            if object['name'] == selected_object:
                return [(object['name'], object['description'], object['image_path'])]
            
def draw_plot(draw_type, title, filename, data, score_name):

    if draw_type == 'layer_type':
        fig, ax = plt.subplots(figsize=(25, 18))
        bar_width = 0.35
        obejct_names = data[score_name]['Encoder'].keys()
        index = np.arange(len(obejct_names))

        bar1 = ax.bar(index, data[score_name]['Encoder'].values(), bar_width, label='Encoder')
        bar2 = ax.bar(index + bar_width, data[score_name]['Decoder'].values(), bar_width, label='Decoder')

        ax.set_xlabel('Object Names', fontsize=24)
        ax.set_ylabel('Score', fontsize=24)
        ax.set_title(title, fontsize=30)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(obejct_names, fontsize=18)  # 글씨 크기 조정
        ax.legend(fontsize=24)

        plt.savefig(filename, dpi=300)
        
    else:
        image_scores = {k: {list(d.keys())[0]: list(d.values())[0] for d in v[score_name]} for k, v in data.items()}
        

        # DataFrame으로 변환 및 정렬
        image_scores_df = pd.DataFrame(image_scores).T
        sorted_columns = sorted(image_scores_df.columns)

        plt.figure(figsize=(15, 8))
        for object_name in image_scores_df.index:
            plt.plot(sorted_columns, image_scores_df.loc[object_name, sorted_columns], marker='o', label=object_name)
        plt.title(title, fontsize=20)
        plt.xlabel(draw_type, fontsize=15)
        plt.ylabel('Score', fontsize=15)
        plt.xticks(fontsize=13)  # X축 눈금 레이블 크기
        plt.yticks(fontsize=13)
        plt.legend()
        plt.grid(True)

        plt.savefig(filename)


        image_scores_mean = image_scores_df.mean()
        
        plt.figure(figsize=(15, 8))
        plt.plot(sorted_columns, image_scores_mean[sorted_columns], marker='o', label='Mean ImageScore')
        plt.title(title, fontsize=20)
        plt.xlabel(draw_type, fontsize=15)
        plt.ylabel('Score', fontsize=15)
        plt.xticks(fontsize=13)  # X축 눈금 레이블 크기
        plt.yticks(fontsize=13)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename[:-4] + "_mean.png")
        
        
        
def draw_path(root_path, draw_type = 'layer_type'):
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    with open('/home/aikusrv01/aiku/t2i_personalization/MasaCtrl/dataset/annotation.json', 'r') as f:
        objects = json.load(f)
    
    if draw_type == 'layer_type':
        target_paths = glob(os.path.join(root_path, '*/masactrl_step4:50_*_1.png'))
        source_paths = glob(os.path.join(root_path, '*/source_step4:50_*_1.png'))
        score = {"CLIPImageScore" : {'Encoder' : {}, 'Decoder' : {}}, "CLIPTextScore" : {'Encoder' : {}, 'Decoder' : {}}}
        for target, source in zip(target_paths, source_paths):
            for object in objects:
                if object['name'] in target:
                    target_prompt = object['description']
                    object_name = object['name']


            s = calculate_clip_s_i2t(target, target_prompt, model_clip, preprocess_clip)
            i = calculate_clip_s(source, target, model_clip, preprocess_clip)
            if 'Encoder' in target:
                score["CLIPImageScore"]['Encoder'][object_name] = i
                score["CLIPTextScore"]['Encoder'][object_name] = s
            else:
                score["CLIPImageScore"]['Decoder'][object_name] = i
                score["CLIPTextScore"]['Decoder'][object_name] = s           

    elif draw_type == 'start_step':
        target_paths = glob(os.path.join(root_path, '*/masactrl_step*:50_Decoder_1.png'))
        source_paths = glob(os.path.join(root_path, '*/source_step*:50_Decoder_1.png'))
        score = {}
        for target, source in zip(target_paths, source_paths):
            for object in objects:
                if object['name'] in target:
                    target_prompt = object['description']
                    object_name = object['name']

            if object_name not in score.keys():
                score[object_name] = {"CLIPImageScore" : [], "CLIPTextScore" : []}

            s = calculate_clip_s_i2t(target, target_prompt, model_clip, preprocess_clip)
            i = calculate_clip_s(source, target, model_clip, preprocess_clip)
            score[object_name]["CLIPImageScore"].append({int(f"{target.split(':')[0].split('step')[-1]}") : i})
            score[object_name]["CLIPTextScore"].append({int(f"{target.split(':')[0].split('step')[-1]}") : s})

    elif draw_type == 'final_step':
        target_paths = glob(os.path.join(root_path, '*/masactrl_step4:*_Decoder_1.png'))
        source_paths = glob(os.path.join(root_path, '*/source_step4:*_Decoder_1.png'))
        score = {}
        
        for target, source in zip(target_paths, source_paths):
            for object in objects:
                if object['name'] in target:
                    target_prompt = object['description']
                    object_name = object['name']

            if object_name not in score.keys():
                score[object_name] = {"CLIPImageScore" : [], "CLIPTextScore" : []}
            s = calculate_clip_s_i2t(target, target_prompt, model_clip, preprocess_clip)
            i = calculate_clip_s(source, target, model_clip, preprocess_clip)
            score[object_name]["CLIPImageScore"].append({int(f"{target.split(':')[-1].split('_')[0]}") : i})
            score[object_name]["CLIPTextScore"].append({int(f"{target.split(':')[-1].split('_')[0]}") : s})

    elif draw_type == 'skip_step_interval':
        target_paths = glob(os.path.join(root_path, '*/masactrl_step4:50_Decoder_?.png'))
        source_paths = glob(os.path.join(root_path, '*/source_step4:50_Decoder_?.png'))
        score = {}
        for target, source in zip(target_paths, source_paths):
            for object in objects:
                if object['name'] in target:
                    target_prompt = object['description']
                    object_name = object['name']

            if object_name not in score.keys():
                score[object_name] = {"CLIPImageScore" : [], "CLIPTextScore" : []}
            s = calculate_clip_s_i2t(target, target_prompt, model_clip, preprocess_clip)
            i = calculate_clip_s(source, target, model_clip, preprocess_clip)
            score[object_name]["CLIPImageScore"].append({int(f"{target.split('_')[-1].split('.')[0]}") : i})
            score[object_name]["CLIPTextScore"].append({int(f"{target.split('_')[-1].split('.')[0]}") : s})

    draw_plot(draw_type, f"CLIP Image Score according to {draw_type}", f"./workdir/{draw_type}_image_score.png", score, "CLIPImageScore")
    draw_plot(draw_type, f"CLIP Text Score according to {draw_type}", f"./workdir/{draw_type}_text_score.png", score, "CLIPTextScore")




if __name__=="__main__":
    torch.cuda.set_device(0)  # set the GPU device

    # Note that you may add your Hugging Face token to get access to the models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model_path = "xyn-ai/anything-v4.0"
    model_path = "CompVis/stable-diffusion-v1-4"
    # model_path = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16).to(device)

    seed = 42
    seed_everything(seed)

    out_dir = "./workdir/all_result/"
    os.makedirs(out_dir, exist_ok=True)

    # source image
    IMAGE_PATH = '/home/aikusrv01/aiku/t2i_personalization/MasaCtrl/dataset/annotation.json'
    dataset = load_dataset(IMAGE_PATH, selected_object = 'pet_cat1')
    source_prompt = ""

    for data in dataset:
        name = data[0]
        source_image = load_image(data[2], device).to(torch.float16)
        
        target_prompt = data[1]
        print(target_prompt)
        prompts = [source_prompt, target_prompt]
        object_dir = os.path.join(out_dir, name)
        os.makedirs(object_dir, exist_ok=True)
        

        # invert the source image
        start_code, latents_list = model.invert(source_image,   # [b, 3, 512, 512]
                                                source_prompt,  # ""
                                                guidance_scale=7.5,
                                                num_inference_steps=50,
                                                return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1) # start_code: [1, 4, 64, 64] => [2, 4, 64, 64]


        # inference the synthesized image with MasaCtrl
        STEP , LAYPER = get_total_step_layer() # 원하는 세팅은 min

    
        for step in STEP:
            for layper in LAYPER:
                editor = MutualSelfAttentionControl(step_idx = step[3], layer_idx=layper[0], total=True)
                regiter_attention_editor_diffusers(model, editor)

                # inference the synthesized image
                image_masactrl = model(prompts,
                                    latents=start_code,
                                    guidance_scale=7.5)

                
                # save the synthesized image
                out_image = torch.cat([source_image * 0.5 + 0.5,
                                    image_masactrl[0:1],
                                    image_masactrl[-1:]], dim=0)
                
                save_image(out_image, os.path.join(object_dir, f"all_step{step[0]}:{step[1]}_{layper[1]}_{step[2]}.png"))
                save_image(out_image[0], os.path.join(object_dir, f"source_step{step[0]}:{step[1]}_{layper[1]}_{step[2]}.png"))
                save_image(out_image[1], os.path.join(object_dir, f"reconstructed_source_step{step[0]}:{step[1]}_{layper[1]}_{step[2]}.png"))
                save_image(out_image[2], os.path.join(object_dir, f"masactrl_step{step[0]}:{step[1]}_{layper[1]}_{step[2]}.png"))
    
    
    target_path = glob('./workdir/min_all_result5/masactrl*')
    source_path = glob('./workdir/min_all_result5/source_step*')
    target_path = sorted(target_path)
    source_path = sorted(source_path)
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    torch.cat([])
    for target, source in zip(target_path, source_path):
        s = calculate_clip_s_i2t(target, target_prompt, model_clip, preprocess_clip)

        i = calculate_clip_s(source, target, model_clip, preprocess_clip)

        print(f"t2t : {s} / i2i : {i}")



    print("Syntheiszed images are saved in", out_dir)
    
    