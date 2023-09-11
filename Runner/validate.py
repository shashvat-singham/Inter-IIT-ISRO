from Analysis.SSIM import ssim
from Analysis.Performance_Measures import psnr

import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import random
torch.manual_seed(10)

def plot_validation(sar_lr, sar_sr, sar_hr, scale_factor, plot_cnt, path):
    sar_lr_plot = F.interpolate(sar_lr, scale_factor=scale_factor, mode='nearest')
    batch_tensor1 = sar_lr_plot
    batch_tensor1 = torch.cat([batch_tensor1, sar_sr], dim=0)
    batch_tensor1 = torch.cat([batch_tensor1, sar_hr], dim=0)
    grid_img = vutils.make_grid(batch_tensor1, nrow=1)
    plt.imshow(grid_img[0].squeeze().cpu().detach().numpy(), "gray")
    plt.savefig(path+'Generated_SAR_epoch_'+str(plot_cnt)+'.png', dpi=350)


def validate(cfg, infer_loader, srun_model, resolution_scale):
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'], exist_ok=True)
    os.makedirs(cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/Validation_Results', exist_ok=True)
    assert resolution_scale == 4 or resolution_scale == 16, "Resolution scale should be 4 or 16"
    print("Current resolution scale: ", resolution_scale)
    device = cfg['DEVICE']
    srun_model = srun_model.to(device)
    running_psnr = 0.
    running_ssim = 0.
    plot_cnt = 0
    for n,(sar_hr, sar_lr) in enumerate(tqdm(infer_loader)):
        if n == len(infer_loader) - 1: break
        sar_hr = sar_hr.to(device)
        sar_lr = sar_lr.to(device)
        batch_size = sar_hr.size(0)

        if resolution_scale == 4:
            res_label = torch.ones(batch_size)*4
            res_label = res_label.unsqueeze(dim=1).to(device) 
            sar_sr, _ = srun_model(sar_lr, res_label)
        elif resolution_scale == 16:
            res_label1 = torch.ones(batch_size)
            res_label1 = res_label1.unsqueeze(dim=1).to(device) 
            res_label2 = torch.ones(batch_size)*4
            res_label2 = res_label2.unsqueeze(dim=1).to(device) 
            sar_sr_mid, _ = srun_model(sar_lr, res_label1)
            sar_sr, _ = srun_model(sar_sr_mid, res_label2)

        ssim_score = ssim(sar_sr, sar_hr, val_range=1, minmax=True)
        psnr_score = psnr(sar_sr, sar_hr, minmax=True)
        running_ssim += ssim_score.item()
        running_psnr += psnr_score.item()

        if random.random() > 0.99 and plot_cnt < 20:
            plot_cnt += 1
            plot_validation(sar_lr, sar_sr, sar_hr, resolution_scale, plot_cnt, cfg['RESULT_DIRS']['GENERATED_IMAGES'] + '/Validation_Results/')
        
    print(' SSIM score: ',round(running_ssim / (len(infer_loader) - 1), 4), \
                                            ' PSNR score: ',round(running_psnr / (len(infer_loader) - 1), 4))