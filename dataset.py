import os
import re
import html
import pandas as pd
import numpy as np
from KoBERT.pretrained_model.tokenization import BertTokenizer
from torch.utils.data.dataset import Dataset


LABEL_DICT = {
    '공포': 0,
    '놀람': 1,
    '분노': 2,
    '슬픔': 3,
    '중립': 4,
    '행복': 5,
    '혐오': 6
}


class Datasets(Dataset):
    """ Adapted from original multimodal transformer code"""

    def __init__(self,
                 data_path,
                 vocab_path,
                 split='train'):
        super(Datasets, self).__init__()
        self.split = split
        self.audio, self.text, self.labels = self.load_data(data_path)
        self.tokenizer, self.vocab = self.load_vocab(vocab_path)

        # special tokens
        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.normalize_string(self.text[idx])
        tokens = self.tokenize(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # ------------------------guideline------------------------------------
        # naming as labels -> use to sampler
        # float32 is required for mfcc function in torchaudio
        # ----------------------------------------------------------------------
        return self.audio[idx].astype(np.float32), tokens, self.labels[idx]

    def tokenize(self, tokens):
        return self.tokenizer.tokenize(tokens)

    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s

    @staticmethod
    def load_data(path):
        data = pd.read_pickle(path)
        text = data['sentence']
        audio = data['audio']
        label = [LABEL_DICT[e] for e in data['emotion']]
        return audio, text, label

    @staticmethod
    def load_vocab(path):
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(
                path, 'pretrained_model', 'vocab.korean.rawtext.list'
            ),
            do_lower_case=False
        )
        return tokenizer, tokenizer.vocab
