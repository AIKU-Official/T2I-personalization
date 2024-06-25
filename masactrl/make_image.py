from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
        
def draw_path(object, root_path, draw_type = 'layer_type'):
    
    
    if draw_type == 'layer_type':
        target_paths = glob(os.path.join(root_path, f'{object}/masactrl_step4:50_*_1.png'))
        source_path = list(glob(os.path.join(root_path, f'{object}/source_step4:50_*_1.png')))[0]

    elif draw_type == 'start_step':
        target_paths = glob(os.path.join(root_path, f'{object}/masactrl_step*:50_Decoder_1.png'))
        source_path = list(glob(os.path.join(root_path, f'{object}/source_step*:50_Decoder_1.png')))[0]
        
    elif draw_type == 'final_step':
        target_paths = glob(os.path.join(root_path, f'{object}/masactrl_step4:*_Decoder_1.png'))
        source_path = list(glob(os.path.join(root_path, f'{object}/source_step4:*_Decoder_1.png')))[0]

    elif draw_type == 'skip_step_interval':
        target_paths = glob(os.path.join(root_path, f'{object}/masactrl_step4:50_Decoder_?.png'))
        source_path = list(glob(os.path.join(root_path, f'{object}/source_step4:50_Decoder_?.png')))[0]

    fig, axes = plt.subplots(1, len(target_paths) + 1, figsize=(30, 5))
    if draw_type == 'start_step':
        target_paths = sorted(target_paths, key=lambda x: int(x.split(':')[-2].split('step')[-1]))
    else:
        target_paths = sorted(target_paths)
    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            image = mpimg.imread(source_path)
            ax.set_title("Source Image")

        else:
            path = target_paths[i-1]
            image = mpimg.imread(path)
            if draw_type == 'layer_type':
                ax.set_title(f"Layer Type: {path.split('_')[-2].split('_')[-1]}")
            elif draw_type == 'start_step':
                ax.set_title(f"Step : {path.split(':')[-2].split('step')[-1]} to 50")
            elif draw_type == 'final_step':
                ax.set_title(f"Step: 4 to {path.split(':')[-1].split('_')[0]}")
            elif draw_type == 'skip_step_interval':
                ax.set_title(f"Skip Step Interval: {path.split('_')[-1].split('.')[0]}")

        ax.imshow(image, cmap='viridis' if i != 0 else None)
        ax.axis('off')
        

    plt.subplots_adjust(wspace=0.0)
    os.makedirs(os.path.join(root_path, 'image'), exist_ok=True)
    plt.savefig(os.path.join(root_path, 'image',object + "_" + draw_type + '.png'))

    
    

if __name__ == "__main__":
    object = 'pet_cat1'
    draw_path(object, './workdir/all_result', 'start_step')
    draw_path(object, './workdir/all_result', 'final_step')
    draw_path(object, './workdir/all_result', 'skip_step_interval')
    draw_path(object, './workdir/all_result', 'layer_type')


    