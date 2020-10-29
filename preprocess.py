import os
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from dataset import LABEL_DICT
from sklearn.model_selection import train_test_split


def allocate_label(sent_id):
    if 0 < sent_id <= 50:
        return "행복"
    elif 50 < sent_id <= 100:
        return "놀람"
    elif 100 < sent_id <= 150:
        return "중립"
    elif 150 < sent_id <= 200:
        return "공포"
    elif 200 < sent_id <= 250:
        return "혐오"
    elif 250 < sent_id <= 300:
        return "분노"
    elif 300 < sent_id <= 350:
        return "슬픔"
    else:
        raise ValueError(f"Invalid sentence id: {sent_id}")


def make_df(lines):
    sentences, emotions = [], []
    cur_label, cur_sent_id = None, 0
    for line in lines:

        # check current label
        line = line.strip()
        if line in LABEL_DICT and len(sentences) % 50 == 0:
            cur_label = line
            continue

        # check the line is valid sentence
        if '[' in line and ']' in line:
            # check label
            cur_sent_id += 1
            cur_expected_label = allocate_label(cur_sent_id)
            assert cur_expected_label == cur_label

            # append sentence & label
            sent = line.replace('[', '').replace(']', '')
            sentences.append(sent)
            emotions.append(cur_label)

    assert len(sentences) == len(emotions)
    return pd.DataFrame({'sentence': sentences, 'emotion': emotions})


def split_df(df, train_size):
    trn_idxs, dev_idxs, tst_idxs = [], [], []
    for label_id in range(7):

        cur_total_idxs = list(range(label_id * 50, (label_id + 1) * 50))
        cur_trn_idxs, cur_eval_idxs = train_test_split(
            cur_total_idxs, test_size=1 - train_size, random_state=42
        )
        cur_dev_idxs, cur_tst_idxs = train_test_split(
            cur_eval_idxs, test_size=.5, random_state=42
        )

        trn_idxs += cur_trn_idxs
        dev_idxs += cur_dev_idxs
        tst_idxs += cur_tst_idxs

        recon_idxs = cur_trn_idxs + cur_dev_idxs + cur_tst_idxs
        assert set(recon_idxs) == set(cur_total_idxs)

    # split dataframe
    train_df = df[df.index.isin(trn_idxs)]
    val_df = df[df.index.isin(dev_idxs)]
    test_df = df[df.index.isin(tst_idxs)]
    return train_df, val_df, test_df


def extract_audio(path, df):
    src_folders = [
        folder for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    src_folders.sort(key=lambda x: int(x))

    for src_folder in tqdm(src_folders, total=len(src_folders)):
        cur_folder_path = os.path.join(path, src_folder)
        src_files = os.listdir(cur_folder_path)
        src_files.sort(key=lambda x: int(x.split('.')[0].split('-')[1]))

        print(src_files)
        print('----------------------------')

    return


def main(args):

    # load script and cleansing
    with open(args.script_path) as f:
        lines = f.readlines()
    total_df = make_df(lines)

    # train-dev-test split
    trn_df, dev_df, tst_df = split_df(total_df, args.train_size)

    # add audio features
    for df, split in zip([trn_df, dev_df, tst_df], ['train', 'dev', 'test']):
        df = extract_audio(args.raw_path, df)
        #df.to_csv(os.path.join(args.save_path, f'{split}.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='./data/korean_multimodal_dataset')
    parser.add_argument('--script_path', type=str, default='./data/korean_multimodal_dataset/script.txt')
    parser.add_argument('--save_path', type=str, default='./data')
    parser.add_argument('--train_size', type=float, default=.8)
    args_ = parser.parse_args()
    main(args_)
