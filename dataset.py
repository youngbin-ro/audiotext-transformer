import os
import re
import html
import torch
import librosa
import logging
import pandas as pd
import numpy as np
from model import load_bert
from torchaudio.transforms import MFCC
from KoBERT.tokenization import BertTokenizer
from torch.utils.data import Dataset, DataLoader


LABEL_DICT = {
    '공포': 0,
    '놀람': 1,
    '분노': 2,
    '슬픔': 3,
    '중립': 4,
    '행복': 5,
    '혐오': 6
}


def get_data_loader(args,
                    data_path,
                    bert_path,
                    num_workers,
                    batch_size=1,
                    split='train'):
    logging.info(f"loading {split} dataset")

    # paths
    data_path = os.path.join(data_path, f'{split}.pkl')
    vocab_path = os.path.join(bert_path, 'vocab.list')
    bert_args_path = os.path.join(bert_path, 'args.bin')

    # MultimodalDataset object
    dataset = MultimodalDataset(
        data_path=data_path,
        vocab_path=vocab_path,
        only_audio=args.only_audio,
        only_text=args.only_text
    )

    # collate_fn
    collate_fn = AudioTextBatchFunction(
        args=args,
        pad_idx=dataset.pad_idx,
        cls_idx=dataset.cls_idx,
        sep_idx=dataset.sep_idx,
        bert_args=torch.load(bert_args_path),
        device='cpu'
    )

    return DataLoader(
        dataset=dataset,
        shuffle=True if split == 'train' else False,
        batch_size=batch_size if split == 'train' else 1,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


class MultimodalDataset(Dataset):
    """ Adapted from original multimodal transformer code"""

    def __init__(self,
                 data_path,
                 vocab_path,
                 only_audio=False,
                 only_text=False):
        super(MultimodalDataset, self).__init__()
        self.only_audio = only_audio
        self.only_text = only_text
        self.use_both = not (self.only_audio or self.only_text)
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
        token_ids = None
        if not self.only_audio:
            tokens = self.normalize_string(self.text[idx])
            tokens = self.tokenize(tokens)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # ------------------------guideline------------------------------------
        # naming as labels -> use to sampler
        # float32 is required for mfcc function in torchaudio
        # ---------------------------------------------------------------------
        return self.audio[idx].astype(np.float32), token_ids, self.labels[idx]

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
        tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
        return tokenizer, tokenizer.vocab


class AudioTextBatchFunction:
    def __init__(self,
                 args,
                 pad_idx,
                 cls_idx,
                 sep_idx,
                 bert_args,
                 device='cpu'):
        self.device = device
        self.only_audio = args.only_audio
        self.only_text = args.only_text
        self.use_both = not (self.only_audio or self.only_text)

        # audio properties
        self.max_len_audio = args.max_len_audio
        self.n_mfcc = args.n_mfcc
        self.n_fft_size = args.n_fft_size
        self.sample_rate = args.sample_rate
        self.resample_rate = args.resample_rate

        # text properties
        self.max_len_bert = bert_args.max_len
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx

        # audio feature extractor
        if not self.only_text:
            self.audio2mfcc = MFCC(
                sample_rate=self.resample_rate,
                n_mfcc=self.n_mfcc,
                log_mels=False,
                melkwargs={'n_fft': self.n_fft_size}
            ).to(self.device)

        # text feature extractor
        if not self.only_audio:
            self.bert = load_bert(args.bert_path, self.device)
            self.bert.eval()

    def __call__(self, batch):
        audios, sentences, labels = list(zip(*batch))
        audio_emb, audio_mask, text_emb, text_mask = None, None, None, None
        with torch.no_grad():

            if not self.only_audio:
                max_len = min(self.max_len_bert, max([len(sent) for sent in sentences]))
                input_ids = torch.tensor([self.pad_with_text(sent, max_len) for sent in sentences])
                text_masks = torch.ones_like(input_ids).masked_fill(input_ids == self.pad_idx, 0).bool()
                text_emb, _ = self.bert(input_ids, text_masks)

            if not self.only_text:
                audio_emb, audio_mask = self.pad_with_mfcc(audios)

        return audio_emb, audio_mask, text_emb, ~text_masks, torch.tensor(labels)

    def _add_special_tokens(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def pad_with_text(self, sentence, max_len):
        sentence = self._add_special_tokens(sentence)
        diff = max_len - len(sentence)
        if diff > 0:
            sentence += [self.pad_idx] * diff
        else:
            sentence = sentence[:max_len - 1] + [self.sep_idx]
        return sentence

    @staticmethod
    def _trim(audio):
        left, right = None, None
        for idx in range(len(audio)):
            if np.float32(0) != np.float32(audio[idx]):
                left = idx
                break
        for idx in reversed(range(len(audio))):
            if np.float32(0) != np.float32(audio[idx]):
                right = idx
                break
        return audio[left:right + 1]

    def pad_with_mfcc(self, audios):
        max_len = min(self.max_len_audio, max([len(audio) for audio in audios]))
        audio_array = torch.zeros(len(audios), self.n_mfcc, max_len).fill_(float('-inf'))
        for idx, audio in enumerate(audios):
            audio = librosa.core.resample(audio, self.sample_rate, self.resample_rate)
            mfcc = self.audio2mfcc(torch.tensor(self._trim(audio)).to(self.device))
            sel_idx = min(mfcc.shape[1], max_len)
            audio_array[idx, :, :sel_idx] = mfcc[:, :sel_idx]

        # (batch_size, n_mfcc, seq_len) -> (batch_size, seq_len, n_mfcc)
        padded = audio_array.transpose(2, 1)
        
        # get key mask
        key_mask = padded[:, :, 0]
        key_mask = key_mask.masked_fill(key_mask != float('-inf'), 0)
        key_mask = key_mask.masked_fill(key_mask == float('-inf'), 1).bool()
        
        # -inf -> 0.0
        padded = padded.masked_fill(padded == float('-inf'), 0.)
        return padded, key_mask
