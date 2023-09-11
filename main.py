from Models.SORTN import SORTN
from Models.SRUN import SRUN
from SRUN_dataset import TMC_optic_dataset, TMC_Dataset, collate_func
import yaml
import argparse

from Runner.train import train
from Runner.inference import inference
from Runner.validate import validate

import torch
from torch import optim
from torch.utils.data import DataLoader
torch.manual_seed(10)

def load_weights(path,optimizer,scheduler,model):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    epoch = ckpt['epoch']
    return optimizer,scheduler,model,epoch
    

parser = argparse.ArgumentParser(description = 'calling for training or performing inference on the model')
parser.add_argument("to_do", help = "Valid Arguments: train, inference")
args = parser.parse_args()
if args.to_do != "train" and args.to_do != "inference" and args.to_do != "validate":
    print('Please enter a valid argument. (train, inference, validate)')
    exit()
to_do = args.to_do

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


srun_model = SRUN(scale_factor=cfg['SCALE_RATIO'], in_channels=1, filter_size=12, num_eram_layers=20)
print('Model parameters: ', sum(p.numel() for p in srun_model.parameters()))
device = cfg['DEVICE']
srun_model = srun_model.to(device)
generator = SORTN()
generator = generator.to(device)
generator.load_state_dict(torch.load(cfg['PRETRAINED_SORTN'])["generator"])

if to_do == 'train':
    
    print('Loading dataset for training')
    dataset = TMC_optic_dataset(cfg, tensor_transform=True)
    a = int(cfg['TRAIN']['TRAIN_TEST_SPLIT'] * len(dataset))
    b = len(dataset) - a
    train_ds, val_ds = torch.utils.data.random_split(dataset, (a, b))
    trainloader = DataLoader(train_ds, batch_size=cfg['TRAIN']['BATCH_SIZE'], collate_fn=collate_func, num_workers=8, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=cfg['TRAIN']['BATCH_SIZE'], collate_fn=collate_func, num_workers=8, shuffle=False)
    trainloader_plot = DataLoader(train_ds, batch_size=1, collate_fn=collate_func, num_workers=8, shuffle=True)
    valloader_plot = DataLoader(val_ds, batch_size=1, collate_fn=collate_func, num_workers=8, shuffle=False)
    
    if cfg['TRAIN']['START_FROM_PRETRAINED_WEIGHTS'] == True:
        
        initial_lr = cfg['TRAIN']['INITIAL_LR']
        beta_1 = cfg['TRAIN']['BETA_1']
        beta_2 = cfg['TRAIN']['BETA_2']
        optimizer = optim.Adam(srun_model.parameters(),lr=initial_lr,betas=(beta_1,beta_2))
        decay_factor = cfg['TRAIN']['DECAY_FACTOR']
        decay_patience = cfg['TRAIN']['PATIENCE']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=decay_factor,patience=decay_patience,verbose=True)
        
        print('Loading pretrained weights for training')
        optimizer,scheduler,srun_model,last_epoch = load_weights(cfg['TRAIN']['PRETRAINED_WEIGHTS'],optimizer,scheduler,srun_model)
        start_epoch = last_epoch + 1
        print('Starting training for epoch ', start_epoch)
        
    else:
        print('Training the model from scratch')
        start_epoch = 0
        
        initial_lr = cfg['TRAIN']['INITIAL_LR']
        beta_1 = cfg['TRAIN']['BETA_1']
        beta_2 = cfg['TRAIN']['BETA_2']
        optimizer = optim.Adam(srun_model.parameters(),lr=initial_lr,betas=(beta_1,beta_2))
        decay_factor = cfg['TRAIN']['DECAY_FACTOR']
        decay_patience = cfg['TRAIN']['PATIENCE']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=decay_factor,patience=decay_patience,verbose=True)
        
    train(cfg, trainloader, valloader, trainloader_plot, valloader_plot, srun_model, generator, optimizer, scheduler, start_epoch)
    
elif to_do == 'validate':
    print('Loading dataset for validation')
    dataset = TMC_Dataset(cfg, tensor_transform=True)
    infer_loader = DataLoader(dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=8, shuffle=False)
    weights = torch.load(cfg['BEST_CKPT'])
    srun_model.load_state_dict(weights['model'])
    resolution_scale = cfg['FINAL_RESOLUTION']
    validate(cfg, infer_loader, srun_model, resolution_scale)

else:
    print('Loading the model for inference')
    weights = torch.load(cfg['BEST_CKPT'])
    srun_model.load_state_dict(weights['model'])
    inference(cfg, srun_model)