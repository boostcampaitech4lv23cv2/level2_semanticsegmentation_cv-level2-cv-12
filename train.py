import os
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import *
from torch import cuda
from utils.utils import label_accuracy_score, add_hist, set_seed
from dataloader import CustomDataLoader, do_transform, collate_fn
from modules.model import create_model
from modules.losses import create_criterion
from modules.scheduler import create_scheduler

import wandb
from utils.set_wandb import wandb_init, finish

import math
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser

import pandas as pd

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='./saved')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--train_json_path', type=str, default='../data/train.json')
    parser.add_argument('--val_json_path', type=str, default='../data/val.json')

    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--use_model', type=str, default='efficient_unet')
    parser.add_argument('--use_losses', type=str, default='cross_entropy')
    parser.add_argument('--use_scheduler', type=str, default='cosine_annealing')

    parser.add_argument('--train_augtype', type=str, default='None')
    parser.add_argument('--save_submission', action='store_true')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--use_amp', action='store_true')

    parser.add_argument('--experiment_name', '-en', type=str, default='segment')
    parser.add_argument('--seed', type=int, default=21)

    args = parser.parse_args()

    return args


def save_model(model, saved_dir, file_name='segment'):
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(args, model):
    print(f'Start training..')
    n_class = 11
    best_mIoU = 0

    # --Loss function 정의
    criterion = create_criterion(args.use_losses)

    # --Optimizer 정의
    optimizer = Adam(params = model.parameters(), lr = args.learning_rate, weight_decay=1e-6)
    schedular = create_scheduler(optimizer, args.use_scheduler)
        
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    # --Dataset 및 DataLoader 설정
    train_dataset = CustomDataLoader(data_dir=args.train_json_path,
                                     mode='train',
                                     transform=do_transform(mode='train', augtype=args.train_augtype),
                                     data_path=args.data_dir)
    val_dataset = CustomDataLoader(data_dir=args.val_json_path,
                                   mode='val',
                                   transform=do_transform(mode='val'),
                                   data_path=args.data_dir)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=collate_fn)

    num_batches = math.ceil(len(train_dataset) / args.batch_size)

    for epoch in range(args.epochs):
        print('\n ### epoch {} ###'.format(epoch+1))
        print(f' Start train #{epoch+1}')
        model.train()

        hist = np.zeros((n_class, n_class))
        with tqdm(total=num_batches) as pbar:
            for step, (images, masks, _) in enumerate(train_loader):
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(args.device), masks.to(args.device)
                
                # device 할당
                model = model.to(args.device)
                
                if args.use_amp:
                    # use FP16
                    with torch.cuda.amp.autocast(enabled=True):
                        # inference
                        outputs = model(images)
                        
                        # loss 계산 (cross entropy loss)
                        loss = criterion(outputs, masks)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

                pbar.update(1)
                tmp_dict = {
                    'Loss': round(loss.item(), 4),
                    'mIoU': round(mIoU, 4)
                }
                pbar.set_postfix(tmp_dict)
                
        lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch" : epoch,
            "learning_rate" : lr,
            "Train loss" : loss.item(),
            "Train mIoU" : mIoU
        })
        schedular.step()

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_every == 0:
            avrg_loss, val_mIoU, val_csv = validation(epoch + 1, model, val_loader, criterion, args.device)
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch (mIoU): {epoch + 1}")
                print(f"Save model in {os.path.join(args.save_dir, args.experiment_name)}")
                best_mIoU = val_mIoU
                save_model(model, args.save_dir, file_name=os.path.join(args.experiment_name, f'best_mIoU.pt'))
                # submission.csv로 저장
                if args.save_submission:
                    if not os.path.isdir(os.path.join(args.save_dir, args.experiment_name)):
                        os.mkdir(os.path.join(args.save_dir, args.experiment_name))
                    val_csv.to_csv(os.path.join(args.save_dir, args.experiment_name, 'val_best.csv'), index=False)
                wandb.log({
                    "epoch" : epoch,
                    "best mIoU epoch" : epoch + 1
                })
            

def validation(epoch, model, data_loader, criterion, device):
    print(f'\n Start validation #{epoch}')
    model.eval()
    category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        hist = np.zeros((n_class, n_class))
        submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)
        for step, (images, masks, image_infos) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long()
            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            if args.save_submission:
                for i in range(len(image_infos)):
                    submission = submission.append({"image_id" : image_infos[i]['file_name'], "PredictionString" : ' '.join(str(e) for e in masks[i].flatten())}, 
                                            ignore_index=True)
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]

        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')

        wandb.log({
                "epoch": epoch,
                "Validation loss" : avrg_loss.item(),
                "Validation mIoU" : mIoU
            })
    return avrg_loss, mIoU, submission


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    model = create_model(args.use_model)

    wandb_init(args)

    train(args, model)
    
    finish()