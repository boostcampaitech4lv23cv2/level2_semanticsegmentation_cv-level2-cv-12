import os
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import models
from torch import cuda
from utils import label_accuracy_score, add_hist, set_seed
from dataloader import CustomDataLoader, do_transform, collate_fn

import wandb
from set_wandb import wandb_init

import math
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='./saved')
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='../data')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)

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
    best_loss = 9999999
    best_mIoU = 0

    # --Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # --Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate, weight_decay=1e-6)

    # --Dataset 및 DataLoader 설정
    train_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'train.json'),
                                     mode='train',
                                     transform=do_transform(mode='train'),
                                     data_path=args.data_dir)
    val_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'val.json'),
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
                
                # inference
                outputs = model(images)['out']
                
                # loss 계산 (cross entropy loss)
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

                # step 주기에 따른 loss 출력
                # if (step + 1) % 25 == 0:
                #     print(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], \
                #             Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
             
        lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "learning_rate" : lr,
            "Train loss" : loss.item(),
            "Train mIoU" : mIoU
        })
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_every == 0:
            avrg_loss, val_mIoU = validation(epoch + 1, model, val_loader, criterion, args.device)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch (loss): {epoch + 1}")
                print(f"Save model in {os.path.join(args.save_dir, args.experiment_name)}")
                best_loss = avrg_loss
                save_model(model, args.save_dir, file_name=os.path.join(args.experiment_name, f'best_loss.pt'))
                wandb.log({
                    "best loss epoch" : epoch + 1
                })
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch (mIoU): {epoch + 1}")
                print(f"Save model in {os.path.join(args.save_dir, args.experiment_name)}")
                best_mIoU = val_mIoU
                save_model(model, args.save_dir, file_name=os.path.join(args.experiment_name, f'best_mIoU.pt'))
                wandb.log({
                    "best mIoU epoch" : epoch + 1
                })
            

def validation(epoch, model, data_loader, criterion, device):
    print(f'\n Start validation #{epoch}')
    model.eval()
    category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
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
        
    return avrg_loss, mIoU


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    model = models.segmentation.fcn_resnet50(pretrained=True)
    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    wandb_init(args)

    train(args, model)
