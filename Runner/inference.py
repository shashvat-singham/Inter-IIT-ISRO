import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob
import os
import matplotlib.pyplot as plt

from Preprocessing.preprocess import *


def inference(cfg, srun_model):
    sar_path_dir = cfg['INFERENCE_SAR_DIR']
    os.makedirs(cfg['INFERENCE_SAVE_DIR'], exist_ok=True)
    sar_paths = sorted(glob(sar_path_dir + '/*.jpeg'))
    for sar_path in sar_paths:
        save_path = cfg['INFERENCE_SAVE_DIR'] + '/' + os.path.basename(sar_path)
        sar_image = cv2.imread(sar_path, 0)
        sar_image = sar_image.astype(float) / 255

        initial_width, initial_height = sar_image.shape[0], sar_image.shape[1]
        scale_ratio = cfg['FINAL_RESOLUTION']
        patch_size = int(cfg['INFERENCE']['PATCH_SIZE'] // scale_ratio)
        
        patches, whole_width, whole_height = crop(sar_image, patch_size)
        print('Total number of patches: ', len(patches))
        device = cfg['DEVICE']
        resolution_scale = cfg['FINAL_RESOLUTION']
        srun_model = srun_model.to(device)
        
        super_resolved = []
        for patch in tqdm(patches):
            tensor_patch = torch.Tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
            batch_size = 1
            with torch.no_grad():
                if resolution_scale == 4:
                    res_label = torch.ones(batch_size)*4
                    res_label = res_label.unsqueeze(dim=1).to(device) 
                    out, _ = srun_model(tensor_patch, res_label)
                elif resolution_scale == 16:
                    res_label1 = torch.ones(batch_size)
                    res_label1 = res_label1.unsqueeze(dim=1).to(device) 
                    res_label2 = torch.ones(batch_size)*4
                    res_label2 = res_label2.unsqueeze(dim=1).to(device) 
                    sar_sr_mid, _ = srun_model(tensor_patch, res_label1)
                    out, _ = srun_model(sar_sr_mid, res_label2)
            out_img = out.cpu().detach().numpy().squeeze()
            super_resolved.append(out_img)
            
        super_resolved_stitched = stitch_patches(super_resolved, whole_width*scale_ratio, whole_height*scale_ratio)
        super_resolved_stitched = super_resolved_stitched[:initial_width*scale_ratio, :initial_height*scale_ratio]
        # in the previous line, we crop out the padding we added earlier in crop function so that we were able to divide the whole image into patches without leaving any remains
        
        super_resolved_stitched = super_resolved_stitched*255
        # super_resolved_stitched = minmax_scale(super_resolved_stitched,0.,255.)
        cv2.imwrite(save_path, super_resolved_stitched)
        print('Final size of super resolved image: ', super_resolved_stitched.shape[0], super_resolved_stitched.shape[1])