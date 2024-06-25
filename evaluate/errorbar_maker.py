import json
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from masactrl.evaluate_clip import calculate_clip_s_i2t, calculate_clip_s

def draw_plot(title, filename, data):

   
    # Extract the data and calculate the means
    object_keys = list(data.keys())
    keys =sorted([list(d.keys())[0] for d in data[object_keys[0]]['CLIPImageScore']])
    x_means = {key: [] for key in keys}
    y_means = {key: [] for key in keys}
    x_vals_all = {key: [] for key in keys}
    y_vals_all = {key: [] for key in keys}

    plt.figure(figsize=(10, 6))
    marker_kinds = ['s', 'D', '*', 'P', 'X', '^', 'v', '<', '>', 'h']
    colors_kinds = ['r', 'g', 'b', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'gray']

    
    markers = dict(zip(object_keys, marker_kinds[:len(object_keys)])) # Different unique markers for each object
    colors = dict(zip(keys, colors_kinds[:len(keys)]))  # Different colors for a, b, c, d
    mean_marker = 'o'  # Marker for means

    # Collect data points for each key across all objects
    for obj in data:
        image_scores = sorted(data[obj]["CLIPImageScore"], key=lambda x: list(x.keys())[0])
        text_scores = sorted(data[obj]["CLIPTextScore"], key=lambda x: list(x.keys())[0])
        
        for key in keys:

            x_vals = [score[key] for score in text_scores if list(score.keys())[0] == key]
            y_vals = [score[key] for score in image_scores if list(score.keys())[0] == key]
            
            x_vals_all[key].extend(x_vals)
            y_vals_all[key].extend(y_vals)
            
            # Plot individual points
            plt.scatter(x_vals, y_vals, edgecolor=colors[key], facecolor='none', label=f'{obj}', marker=markers[obj])

    # Calculate and plot means with error bars for each key
    for key in keys:
        x_mean = np.mean(x_vals_all[key])
        y_mean = np.mean(y_vals_all[key])
        
        x_means[key].append(x_mean)
        y_means[key].append(y_mean)
        
        x_err = np.std(x_vals_all[key]) / 2
        y_err = np.std(y_vals_all[key]) / 2
        
        plt.errorbar(x_mean, y_mean, xerr=x_err, yerr=y_err, fmt=mean_marker, ecolor=colors[key], mfc=colors[key], mec=colors[key], label=f'{key} Mean', capsize=5)

    # Adding labels and title
    plt.xlim(0.15, 0.35)
    plt.ylim(0.5, 1.0)
    plt.xlabel('Text Alignment Score')
    plt.ylabel('Image Alignment Score')
    plt.title(title)

    # Create a separate legend for markers and colors
    # Create a separate legend for markers and colors
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    marker_handles = [plt.Line2D([0], [0], color='w', marker=markers[obj], markerfacecolor='none', markeredgecolor='k', label=obj) for obj in markers]
    color_handles = [plt.Line2D([0], [0], color=colors[key], marker='o', linestyle='None', label=f'{key}') for key in colors]
    mean_handle = plt.Line2D([0], [0], color='k', marker=mean_marker, linestyle='None', markerfacecolor='k', label='Mean')

    # Combine handles and labels
    handles_combined = marker_handles + color_handles + [mean_handle]

    # Plot the legend outside the plot
    plt.legend(handles=handles_combined, bbox_to_anchor=(1.05, 1), loc='upper left', title='Legend')

    # Adjust the plot to make space for the legend
    plt.subplots_adjust(right=0.75)

    # Show the plot
    plt.savefig(filename)

def draw_path(root_path, draw_type = 'layer_type'):
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    with open('/home/aikusrv01/aiku/t2i_personalization/MasaCtrl/dataset/annotation.json', 'r') as f:
        objects = json.load(f)         

    if draw_type == 'start_step':
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

    
    draw_plot(f"{draw_type}", f"./workdir/a/{draw_type}_plot.png", score)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    os.makedirs("./workdir/a", exist_ok=True)
    draw_path('./workdir/all_result', 'start_step')
    draw_path('./workdir/all_result', 'final_step')
    draw_path('./workdir/all_result', 'skip_step_interval')