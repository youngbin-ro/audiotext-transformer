import os
import re
import html
import torch
import pandas as pd
import numpy as np
from KoBERT.pretrained_model.tokenization import BertTokenizer
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)


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
                    batch_size,
                    num_workers,
                    split='train'):

    # paths
    data_path = os.path.join(data_path, f'{split}.pkl')
    vocab_path = os.path.join(bert_path, 'vocab.korean.rawtext.list')
    bert_args_path = os.path.join(bert_path, 'training_args.bin')

    # MultimodalDataset object
    dataset = MultimodalDataset(
        data_path=data_path,
        vocab_path=vocab_path,
        split=split
    )

    # sampler
    if split == 'train':
        sampler = RandomSampler(dataset)
    elif split == 'dev' or split == 'test':
        sampler = SequentialSampler(dataset)
    else:
        raise ValueError(f"Please check your data split: {split}")

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
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


class MultimodalDataset(Dataset):
    """ Adapted from original multimodal transformer code"""

    def __init__(self,
                 data_path,
                 vocab_path,
                 split='train'):
        super(MultimodalDataset, self).__init__()
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
        self.only_audio = args.do_audio
        self.device = device

        # related to audio--------------------
        self.max_len_a = args.max_len_for_audio
        self.n_mfcc = args.n_mfcc
        self.n_fft_size = args.n_fft_size
        self.sample_lr = args.sample_rate
        self.resample_lr = args.resample_rate

        self.audio2mfcc = torchaudio.transforms.MFCC(sample_rate=self.resample_lr,
                                                     n_mfcc=self.n_mfcc,
                                                     log_mels=False,
                                                     melkwargs={'n_fft': self.n_fft_size}).to(self.device)

        if not self.only_audio:
            # related to text--------------------
            self.max_len_t = bert_args.max_len
            self.pad_idx = pad_idx
            self.cls_idx = cls_idx
            self.sep_idx = sep_idx

            self.bert_config = BertConfig(args.bert_config_path)
            self.bert_config.num_labels = num_label_from_bert

            self.model = BertForTextRepresentation(self.bert_config).to(self.device)
            pretrained_weights = torch.load(args.bert_model_path
                                            , map_location=torch.device(self.device))
            self.model.load_state_dict(pretrained_weights, strict=False)
            self.model.eval()

    def __call__(self, batch):
        audio, texts, label = list(zip(*batch))

        if not self.only_audio:
            # Get max length from batch
            max_len = min(self.max_len_t, max([len(i) for i in texts]))
            texts = torch.tensor(
                [self.pad_with_text([self.cls_idx] + text + [self.sep_idx], max_len) for text in texts])
            masks = torch.ones_like(texts).masked_fill(texts == self.pad_idx, 0)

            with torch.no_grad():
                # text_emb = last layer
                text_emb, cls_token = self.model(**{'input_ids': texts.to(self.device),
                                                    'attention_mask': masks.to(self.device)})

                audio_emb, audio_mask = self.pad_with_mfcc(audio)

            return audio_emb, audio_mask, text_emb, torch.tensor(label)
        else:
            audio_emb, audio_mask = self.pad_with_mfcc(audio)
            return audio_emb, audio_mask, None, torch.tensor(label)

    def pad_with_text(self, sample, max_len):
        diff = max_len - len(sample)
        if diff > 0:
            sample += [self.pad_idx] * diff
        else:
            sample = sample[-max_len:]
        return sample

    def pad_with_mfcc(self, audios):
        max_len_batch = min(self.max_len_a, max([len(a) for a in audios]))
        audio_array = torch.zeros(len(audios), self.n_mfcc, max_len_batch).fill_(float('-inf')).to(self.device)
        for ix, audio in enumerate(audios):
            audio_ = librosa.core.resample(audio, self.sample_lr, self.resample_lr)
            audio_ = torch.tensor(self.trimmer(audio_))
            mfcc = self.audio2mfcc(audio_.to(self.device))
            sel_ix = min(mfcc.shape[1], max_len_batch)
            audio_array[ix, :, :sel_ix] = mfcc[:, :sel_ix]
        # (bat, n_mfcc, seq) -> (bat, seq, n_mfcc)
        padded_array = audio_array.transpose(2, 1)

        # key masking
        # (batch, seq)
        key_mask = padded_array[:, :, 0]
        key_mask = key_mask.masked_fill(key_mask != float('-inf'), 0).masked_fill(key_mask == float('-inf'), 1).bool()

        # -inf -> 0.0
        padded_array = padded_array.masked_fill(padded_array == float('-inf'), float(0))
        return padded_array, key_mask

    def trimmer(self, audio):
        fwd_audio = []
        fwd_init = np.float32(0)
        for a in audio:
            if fwd_init != np.float32(a):
                fwd_audio.append(a)

        bwd_init = np.float32(0)
        bwd_audio = []
        for a in fwd_audio[::-1]:
            if bwd_init != np.float32(a):
                bwd_audio.append(a)
        return bwd_audio[::-1]
