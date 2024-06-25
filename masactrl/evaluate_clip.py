import os
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from glob import glob

from transformers import CLIPProcessor, CLIPModel

def calculate_clip_s_i2t(image_path, text, model_clip, preprocess_clip):
    image = Image.open(image_path)
    #image = preprocess_clip(image).unsqueeze(0).to('cuda')
    image = preprocess_clip(images=image, return_tensors='pt')['pixel_values'].to('cuda')
    text = preprocess_clip(text, return_tensors='pt')['input_ids'].to('cuda')
    with torch.no_grad():
        # image_features = model_clip.encode_image(image)
        # text_features = model_clip.encode_text(text)
        image_features = model_clip.get_image_features(pixel_values=image)
        text_features = model_clip.get_text_features(input_ids=text)
    s = torch.cosine_similarity(image_features, text_features, dim=-1)
    return s.item()

def calculate_clip_s(origial_image_path, generated_image_path, model_clip, preprocess_clip):
    original_image = Image.open(origial_image_path)
    generated_image = Image.open(generated_image_path)
    #original_image = preprocess_clip(original_image).unsqueeze(0).to('cuda')
    #generated_image = preprocess_clip(generated_image).unsqueeze(0).to('cuda')
    original_image = preprocess_clip(images=original_image, return_tensors='pt')['pixel_values'].to('cuda')
    generated_image = preprocess_clip(images=generated_image, return_tensors='pt')['pixel_values'].to('cuda')
    with torch.no_grad():
        # original_image_features = model_clip.encode_image(original_image)
        # generated_image_features = model_clip.encode_image(generated_image)
        original_image_features = model_clip.get_image_features(pixel_values=original_image)
        generated_image_features = model_clip.get_image_features(pixel_values=generated_image)
    s = torch.cosine_similarity(original_image_features, generated_image_features, dim=-1)
    return s.item()

def calculate_clip_s_for_folder(original_image_folder, generated_image_folder):
    s_list = []
    file_list = glob(os.path.join(generated_image_folder, '*.png'))
    file_list += glob(os.path.join(generated_image_folder, '*.jpg'))
    #model_clip, _,  preprocess_clip = open_clip.create_model_and_transforms('ViT-L-14', device='cuda')
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    for i, file in enumerate(tqdm(file_list)):
        file = os.path.basename(file)
        original_image_path = os.path.join(original_image_folder, file)
        generated_image_path = os.path.join(generated_image_folder, file)
        s = calculate_clip_s(original_image_path, generated_image_path, model_clip, preprocess_clip)
        s_list.append(s)
    if len(s_list) == 0:
        return 0
    print("clip score: ", sum(s_list) / len(s_list))
    return sum(s_list) / len(s_list)

if __name__ == '__main__':
    paths = glob('./workdir/min_all_result/4_*_10_16/masactrl.png')
    text =  'a photo of a running dog'
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    s = {path.split("/")[-2] : calculate_clip_s_i2t(path, text, model_clip, preprocess_clip) for path in paths}
    
    print(sorted(s.items(), key=lambda item: item[1], reverse=True))