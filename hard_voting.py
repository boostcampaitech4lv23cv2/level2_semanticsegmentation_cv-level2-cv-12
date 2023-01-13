import pandas as pd
from tqdm import tqdm
import argparse
import os


'''
csv 파일을 저장할 때, Test score가 제일 잘나온 csv를 1_output.csv로 저장.
그 다음 csv를 2_output.csv로 저장하기.
'''


def parser_agrs():
    parser = argparse.ArgumentParser(description='hard voting')
    parser.add_argument('--csv-folder-path', '-f', type=str, help='csv 파일이 저장된 폴더', default='/opt/ml/input/code/ensemble')
    parser.add_argument('--save-path', '-s', type=str, help='ensemble csv file path', default='/opt/ml/input/code/ensemble')

    args = parser.parse_args()
    return args


def hard_voting(args):
    submission_csvs = os.listdir(args.csv_folder_path)
    submission_csvs.sort()

    df_list = list()
    for submission_csv in submission_csvs:
        df_list.append(pd.read_csv(os.path.join(args.csv_folder_path, submission_csv)))

    ensemble_sub = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)

    for img_id in tqdm(range(len(df_list[0]))):
        mask_list = list()
        for idx in range(len(df_list)):
            mask_list.append(df_list[idx]['PredictionString'][img_id].split())

        result_str = list()

        # 픽셀별로 여러가지 csv에 대해 voting 시작
        for x in range(len(mask_list[0])):  # 65536
            pixel_voting = {'0' : 0, '1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0,
                            '6' : 0, '7' : 0, '8' : 0, '9' : 0, '10' : 0}

            for y in range(len(mask_list)): # csv 파일 갯수
                pixel_voting[mask_list[y][x]] += 1

            # 제일 많이 voting된 catergory를 선택
            pixel_voting = [k for k, v in pixel_voting.items() if v == max(pixel_voting.values())]

            # voting 결과가 1개인 경우
            if len(pixel_voting) == 1:
                result_str.append(pixel_voting[0])
            # voting 결과가 여러개인 경우 제일 결과가 좋은 csv를 기준으로 판단
            else:
                for i in range(len(mask_list)):
                    best_csv_pixel = mask_list[i][x]
                    if best_csv_pixel in pixel_voting:
                        result_str.append(best_csv_pixel)
                        break

        ensemble_sub = ensemble_sub.append({
            'image_id':df_list[0]['image_id'][img_id],
            'PredictionString':' '.join(str(i) for i in result_str)
            },
            ignore_index=True)

    ensemble_sub.to_csv(os.path.join(args.save_path, 'ensemble_submission.csv'), index=False)


if __name__ == "__main__":
    args = parser_agrs()
    hard_voting(args)