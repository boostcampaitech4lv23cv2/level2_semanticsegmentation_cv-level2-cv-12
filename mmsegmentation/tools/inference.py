# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
import json
import pandas as pd
import numpy as np
import albumentations as A
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmcv.parallel import MMDataParallel

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None

    test_dataset = build_dataset(cfg.data.test)
    test_data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    # model.PALETTE = checkpoint['meta']['PALETTE']
    # model = MMDataParallel(model.cuda(), device_ids=[0])
    model = build_dp(model, device_ids=[0])
    result = single_gpu_test(model, test_data_loader)

    # make csv file
    print('\n Saving submission.csv ...')
    submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)
    json_dir = os.path.join('/opt/ml/input/data/test.json')
    with open(json_dir, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # input_size = 512
    result_size = 256
    transformed = A.Compose([A.Resize(result_size, result_size)])

    for image_id, pred in enumerate(result):
        temp_mask = list()
        image_id = datas['images'][image_id]
        file_name = image_id['file_name']
        
        mask = np.array(pred, dtype='uint8')
        mask = transformed(image=mask)
        temp_mask.append(mask['image'])

        tmn = np.array(temp_mask)
        tmn = tmn.reshape([tmn.shape[0], result_size*result_size]).astype(int)

        tmn_flat = tmn.flatten()
        submission = submission.append({
            'image_id':file_name, 
            "PredictionString":' '.join(str(i) for i in tmn_flat.tolist())
            },
            ignore_index=True)

    submission.to_csv(os.path.join(args.work_dir, 'submission.csv'), index=False)
    print('Saved submission.csv !!!')


if __name__ == '__main__':
    main()
