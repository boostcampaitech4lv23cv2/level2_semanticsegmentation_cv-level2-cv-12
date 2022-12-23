import streamlit as st
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
import numpy as np
import math
import torch
from copy import deepcopy
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import webcolors
from matplotlib.patches import Patch
import albumentations as A

import sys
sys.path.append('/opt/ml/level2_semanticsegmentation_cv-level2-cv-12')

from dataloader import CustomDataLoader, do_transform, collate_fn
from utils.utils import label_to_color_image, _fast_hist, label_accuracy_score

class_colormap = pd.read_csv("../class_dict.csv")

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title="annotation", layout="wide")

## ex) streamlit run app.py --server.port=30001 -- --submission_csv <path> --gt_json <path> --dataset_path <path>
parser = argparse.ArgumentParser(description='basic Argparse')
parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/', help='데이터셋 폴더 경로 ex)/opt/ml/dataset/')
parser.add_argument('--train_gt', type=str, default='train.json', help='train dataset json 파일의 이름 ex)train.json')
parser.add_argument('--valid_gt', type=str, default='val.json', help='valid dataset json 파일의 이름 ex)valid.json')
parser.add_argument('--valid_csv', type=str, default='/opt/ml/level2_semanticsegmentation_cv-level2-cv-12/submission/baseline/submission.csv', help='output csv 파일 경로 ex)/opt/ml/submission.csv')
args = parser.parse_args()

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def draw_gt_mask(h, w, cats, anns, category_names, coco):
    masks = np.zeros((h, w))
    masks = masks.astype(np.int8)
    anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
    for i in range(len(anns)):
        className = get_classname(anns[i]['category_id'], cats)
        pixel_value = category_names.index(className)
        masks[coco.annToMask(anns[i]) == 1] = pixel_value
    masks = masks.astype(np.int8)
    return masks

def cal_IoU(gt_mask, pred_mask):
    hist = _fast_hist(gt_mask, pred_mask, 11)
    acc, acc_cls, mean_iu, fwavacc, iu = label_accuracy_score(hist)
    return mean_iu

@st.experimental_singleton
def init_data(_val_data, image_ids, image_filenames, image_infos, pred_df):
    category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    cat_ids = _val_data.getCatIds()
    cats = _val_data.loadCats(cat_ids)
    data = {}
    for i, filename in enumerate(image_filenames):
        if filename not in data:
            data[filename] = {}
        h = image_infos[i]["height"]
        w = image_infos[i]["width"]
        ann_ids = _val_data.getAnnIds(imgIds=image_ids[i])
        anns = _val_data.loadAnns(ann_ids)
        img_path = args.data_dir + image_filenames[i]
        img = cv.imread(img_path)
        img = cv.cvtColor(img ,cv.COLOR_BGR2RGB)
        gt_mask = draw_gt_mask(h, w, cats, anns, category_names, _val_data)
        transform = A.Compose([A.Resize(256, 256)])
        transformed = transform(image=img, mask=gt_mask)
        img = transformed['image']
        data[filename]['image'] = img
        gt_mask = transformed['mask']
        seg_img = label_to_color_image(gt_mask)
        data[filename]['gt_mask']= seg_img
        pred = np.fromstring(pred_df['PredictionString'][i], dtype=int, sep=' ')
        pred = pred.reshape(256 ,256)
        data[filename]['mIoU'] = cal_IoU(gt_mask, pred)
        pred = label_to_color_image(pred)
        data[filename]['pred'] = pred
    return data

def main():
    st.title("Annotation Visualization")
    val_data = COCO(args.data_dir+ args.valid_gt)
    image_ids = val_data.getImgIds()
    image_infos = val_data.loadImgs(image_ids)
    
    all_image_filenames = [img['file_name'] for img in image_infos]
    pred_df = pd.read_csv(args.valid_csv)
    data = init_data(val_data, image_ids, all_image_filenames, image_infos, pred_df)
    side_col1, side_col2, side_col3 = st.sidebar.columns([1,1,3])
    if "img_idx" not in st.session_state:
        st.session_state["img_idx"] = 0
    if side_col1.button('Prev'):
        st.session_state["img_idx"] -= 1
    if side_col2.button('Next'):
        st.session_state["img_idx"] += 1
    if "mIoU" not in st.session_state:
        st.session_state["mIoU"] = (0., 1.)
    st.session_state["mIoU"] = st.sidebar.slider(f'이미지 mIooU 범위', min_value=0., max_value=1., value=st.session_state["mIoU"], step=0.01, key='mIoU slider')
    print(st.session_state["mIoU"])
    image_filenames = [filename for i, filename in enumerate(all_image_filenames)
                    if data[filename]['mIoU'] >= st.session_state["mIoU"][0] and data[filename]['mIoU'] <= st.session_state["mIoU"][1]]
    st.session_state["img_idx"] = st.sidebar.selectbox('Selcet Image', range(len(image_filenames)), format_func=lambda x:image_filenames[x], index=st.session_state["img_idx"])
    filename = image_filenames[st.session_state["img_idx"]]
    st.header(filename)
    cols = st.columns(11)
    color1 = cols[0].color_picker('Backgroud', '#000000')
    color2 = cols[1].color_picker('General trash', '#C00080')
    color3 = cols[2].color_picker('Paper', '#0080C0')
    color4 = cols[3].color_picker('Paper pack', '#008040')
    color5 = cols[4].color_picker('Metal', '#800000')
    color6 = cols[5].color_picker('Glass', '#400080')
    color7 = cols[6].color_picker('Plastic', '#4000C0')
    color8 = cols[7].color_picker('Styrofoam', '#C08040')
    color9 = cols[8].color_picker('Plastic bag', '#C0C080')
    color10 = cols[9].color_picker('Battery', '#404080')
    color11 = cols[10].color_picker('Clothing', '#8000C0')
    col1, col2, col3 = st.columns(3)
    col1.text('Image')
    col2.text('GT')
    col3.text('Prediction')
    col1.image(data[filename]['image'])
    col2.image(data[filename]['gt_mask'])
    col3.image(data[filename]['pred'])
    st.metric("mIoU", data[filename]['mIoU'])
main()