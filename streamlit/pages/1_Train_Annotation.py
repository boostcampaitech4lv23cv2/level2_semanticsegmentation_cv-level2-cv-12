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

import sys
sys.path.append('/opt/ml/level2_semanticsegmentation_cv-level2-cv-12')

from dataloader import CustomDataLoader, do_transform, collate_fn
from utils.utils import label_to_color_image

class_colormap = pd.read_csv("../class_dict.csv")

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title="annotation", layout="wide")

## ex) streamlit run app.py --server.port=30001 -- --submission_csv <path> --gt_json <path> --dataset_path <path>
parser = argparse.ArgumentParser(description='basic Argparse')
# parser.add_argument('--gt_json', type=str, default='/opt/ml/input/data/dataset/ufo/annotation.json', help='Ground Truth 데이터의 json 파일 경로 ex)/opt/ml/dataset/train.json')
parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/', help='데이터셋 폴더 경로 ex)/opt/ml/input/data/')
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

def main():
    st.title("Annotation Visualization")
    category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    train_data = COCO(args.data_dir+ args.train_gt)
    image_ids = train_data.getImgIds()
    image_infos = train_data.loadImgs(image_ids)
    image_filenames = [img['file_name'] for img in image_infos]

    cat_ids = train_data.getCatIds()
    cats = train_data.loadCats(cat_ids)

    side_col1, side_col2, side_col3 = st.sidebar.columns([1,1,3])
    if "img_idx" not in st.session_state:
        st.session_state["img_idx"] = 0
    if side_col1.button('Prev'):
        st.session_state["img_idx"] -= 1
    if side_col2.button('Next'):
        st.session_state["img_idx"] += 1
    st.session_state["img_idx"] = st.sidebar.selectbox('Selcet Image', range(len(image_filenames)), format_func=lambda x:image_filenames[x], index=st.session_state["img_idx"])
    img_path = args.data_dir + image_filenames[st.session_state["img_idx"]]
    img = cv.imread(img_path)
    img = cv.cvtColor(img ,cv.COLOR_BGR2RGB)

    ann_ids = train_data.getAnnIds(imgIds=image_ids[st.session_state["img_idx"]])
    anns = train_data.loadAnns(ann_ids)

    h = image_infos[st.session_state["img_idx"]]["height"]
    w = image_infos[st.session_state["img_idx"]]["width"]
    masks = draw_gt_mask(h, w, cats, anns, category_names, train_data)
    seg_img = label_to_color_image(masks)
    
    st.header(image_filenames[st.session_state["img_idx"]])
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
    col1, col2 = st.columns(2)
    col1.image(img)
    col2.image(seg_img)

main()