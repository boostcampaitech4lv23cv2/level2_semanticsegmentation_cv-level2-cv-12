from urllib.parse import ParseResultBytes
from PIL import Image
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import shutil
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/opt/ml/input/data", type=str,
                        help="coco dataset directory")
    parser.add_argument("--file_name", default="train", type=str,
                        help="annotation file name")
    return parser.parse_args()


def save_colored_mask(mask, save_path):
    mask_img = Image.fromarray(mask.astype(np.uint8)) 
    mask_img.save(save_path)


def main(args):
    annotation_file = os.path.join(args.input_dir, f'{args.file_name}.json')
    os.makedirs(os.path.join(args.input_dir, f'mmseg/{args.file_name}/images'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, f'mmseg/{args.file_name}/annotations'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for index, imgId in tqdm.tqdm(enumerate(imgIds), total=len(imgIds)):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)
        anns_img = np.zeros((img['height'], img['width']))
        if len(annIds) > 0:
            for ann in anns:
                anns_img = np.maximum(anns_img, coco.annToMask(ann)*ann['category_id'])
            img_origin_path = os.path.join(args.input_dir, img['file_name'])
            img_output_path = os.path.join(args.input_dir, f'mmseg/{args.file_name}/images', f'{index:04}.jpg')
            seg_output_path = os.path.join(args.input_dir, f'mmseg/{args.file_name}/annotations', f'{index:04}.png')
            shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(anns_img, seg_output_path)


if __name__ == '__main__':
    args = get_args()
    main(args)