import argparse
import torch
import logging
from utils import set_seed
from dataset import get_data_loader


def main(args):
    set_seed(args.seed)

    # load data
    loaders = (get_data_loader(
        args=args,
        data_path=args.data_path,
        bert_path=args.bert_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=split
    ) for split in ['train', 'dev'])
    trn_loader, dev_loader = loaders

    # initialize model
    mult = MultimodalTransformer(
        # TODO
    ).to(args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--only_audio', action='store_true')
    parser.add_argument('--only_text', action='store_true')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--bert_path', type=str, default='./KoBERT')
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--logging_steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=12)

    # dropouts
    parser.add_argument('--attn_dropout', type=float, default=.2)
    parser.add_argument('--relu_dropout', type=float, default=.1)
    parser.add_argument('--emb_dropout', type=float, default=.2)
    parser.add_argument('--res_dropout', type=float, default=.1)
    parser.add_argument('--out_dropout', type=float, default=.1)

    # architecture
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)

    # training
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--clip', type=float, default=0.8)
    parser.add_argument('--when', type=int, default=10)
    parser.add_argument('--batch_chunk', type=int, default=1)
    parser.add_argument('--warmup_percent', type=float, default=0.1)

    # data processing
    parser.add_argument('--max_len_text', type=int, default=64)
    parser.add_argument('--max_len_audio', type=int, default=400)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--resample_rate', type=int, default=16000)
    parser.add_argument('--n_fft_size', type=int, default=400)
    parser.add_argument('--n_mfcc', type=int, default=40)

    args_ = parser.parse_args()

    # -------------------------------------------------------------- #

    # seed and device setting
    set_seed(args_.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_.device = device

    # log setting
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    main(args_)
