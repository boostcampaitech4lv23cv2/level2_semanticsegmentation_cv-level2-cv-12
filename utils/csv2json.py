import cv2
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pycocotools._mask as _mask

class_labels = {
    # 0: "Background",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}


def parse_args():
    parser = argparse.ArgumentParser(description='csv file to json file')
    parser.add_argument('--csv-path', type=str, help='submission csv file path')

    args = parser.parse_args()
    return args


def decode(mask_file):
    mask_file = mask_file.split()
    img = np.zeros(256*256, dtype=np.uint8)
    for i, m in enumerate(mask_file):
        img[i] = int(m)
    return img.reshape(256, 256)


def make_csv_to_json(args):
    sub_df = pd.read_csv(args.csv_path)

    # with open('/opt/ml/input/data/test.json', 'r') as f:
    #     test_json = json.load(f)

    # # annotation 정보를 넣기 위한 리스트 생성
    # test_json['annotations'] = []

    # ann_id = 0
    # for im_id in tqdm(range(len(sub_df))):
    #     ann_id += 1
    #     if ann_id == 15:
    #         break
    #     # (65536, ) to (256, 256)
    #     mask_info = np.array(sub_df['PredictionString'][im_id].split(), dtype=np.uint16).reshape(256, 256)

    #     img = Image.fromarray(np.uint16(mask_info))
    #     img = img.resize((512, 512))
    #     img = np.array(img, dtype=np.uint8)
        
    #     for category_id, category_name in class_labels.items():
    #         new_img = np.where(img != category_id, 0, img)
    #         ret, binary = cv2.threshold(new_img, 0.5, 255, cv2.THRESH_BINARY)
    #         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for img_id in tqdm(range(len(sub_df))):
        decoded_mask = decode(sub_df['PredictionString'][img_id])
        decoded_mask = cv2.resize(decoded_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        original_img = cv2.imread('/opt/ml/input/data/'+f'{sub_df["image_id"][img_id]}')
        cv2.imwrite('/opt/ml/input/data/mmseg/pseudo/images/'+f'pseudo_{img_id}.jpg', original_img)
        cv2.imwrite('/opt/ml/input/data/mmseg/pseudo/annotations/'+f'pseudo_{img_id}.png', decoded_mask)

        

if __name__ == "__main__":
    args = parse_args()
    make_csv_to_json(args)