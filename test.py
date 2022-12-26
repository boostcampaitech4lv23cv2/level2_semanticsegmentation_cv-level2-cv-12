import os
import torch
import torch.nn as nn
from torch import cuda
from torchvision import models
import albumentations as A
from modules.model import create_model

import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

from dataloader import CustomDataLoader, do_transform, collate_fn


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='./submission')
    parser.add_argument('--model_path', type=str, default='./saved/segment/best_mIoU.pt')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--use_model', type=str, default='efficient_unet')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--resize', type=int, default=256)

    parser.add_argument('--experiment_name', '-en', type=str, default='segment')

    args = parser.parse_args()

    return args


def test(args, model):
    test_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'test.json'),
                                   mode='test',
                                   transform=do_transform(mode='test'),
                                   data_path=args.data_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                collate_fn=collate_fn)
    size = args.resize
    transform = A.Compose([A.Resize(size, size)])
    print('\nStart prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(args.device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])

    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    print("Start Inference")
    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    if not os.path.isdir(os.path.join(args.save_dir, args.experiment_name)):
        os.mkdir(os.path.join(args.save_dir, args.experiment_name))
        
    submission.to_csv(os.path.join(args.save_dir, args.experiment_name, 'submission.csv'), index=False)
    print("End Inference")


if __name__ == "__main__":
    args = parse_args()

    model = create_model(args.use_model)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(args.device)

    test(args, model)