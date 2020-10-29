import argparse
import torch
import logging
import os
from utils import set_seed, get_optimizer_and_scheduler
from dataset import get_data_loader
from model import MultimodalTransformer
from tqdm import tqdm
from eval import evaluate


def train(args,
          model,
          trn_loader,
          optimizer):

    trn_loss, logging_loss = 0, 0
    loss_fct = torch.nn.CrossEntropyLoss()
    iterator = tqdm(enumerate(trn_loader), desc='steps', total=len(trn_loader))

    # start steps
    for step, batch in iterator:
        model.train()

        # unpack and set inputs
        batch = map(lambda x: x.to(args.device), batch)
        audios, texts, labels = batch
        labels = labels.squeeze(-1).long()

        # feed to model and get loss
        logit, hidden = model(audios, texts)
        loss = loss_fct(logit, labels.view(-1))
        trn_loss += loss.item()

        # update the model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        model.zero_grad()
        args.global_step += 1

        # summary
        if args.global_step % args.logging_steps == 0:
            cur_logging_loss = (trn_loss - logging_loss) / args.logging_steps
            logging.info("train loss: {:.4f}".format(cur_logging_loss))
            logging_loss = trn_loss


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
    model = MultimodalTransformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_classes=args.n_classes,
        only_audio=args.only_audio,
        only_text=args.only_text,
        d_audio_orig=args.n_mfcc,
        d_text_orig=768,    # BERT hidden size
        d_model=args.d_model,
        attn_dropout=args.attn_dropout,
        relu_dropout=args.relu_dropout,
        emb_dropout=args.emb_dropout,
        res_dropout=args.res_dropout,
        out_dropout=args.out_dropout,
        attn_mask=args.attn_mask
    ).to(args.device)

    # warmup scheduling
    args.total_steps = round(len(trn_loader) * args.epochs)
    args.warmup_steps = round(args.total_steps * args.warmup_percent)

    # optimizer & scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    # training
    logging.info('training starts')
    model.zero_grad()
    args.global_step = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc='epochs'):
        train(args, model, trn_loader, optimizer)
        loss, f1 = evaluate(model, dev_loader, args.device)
        model_name = "epoch{}-loss{:.4f}-f1{:.4f}.bin".format(epoch, loss, f1)
        model_path = os.path.join(args.save_path, model_name)
        torch.save(model.state_dict(), model_path)
    logging.info('training ended')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--only_audio', action='store_true')
    parser.add_argument('--only_text', action='store_true')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--bert_path', type=str, default='./KoBERT')
    parser.add_argument('--save_path', type=str, default='./result')
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    # dropouts
    parser.add_argument('--attn_dropout', type=float, default=.2)
    parser.add_argument('--relu_dropout', type=float, default=.1)
    parser.add_argument('--emb_dropout', type=float, default=.2)
    parser.add_argument('--res_dropout', type=float, default=.1)
    parser.add_argument('--out_dropout', type=float, default=.1)

    # architecture
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=40)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--attn_mask', action='store_false')

    # training
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--warmup_percent', type=float, default=0.1)

    # data processing
    parser.add_argument('--max_len_audio', type=int, default=400)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--resample_rate', type=int, default=16000)
    parser.add_argument('--n_fft_size', type=int, default=400)
    parser.add_argument('--n_mfcc', type=int, default=64)

    args_ = parser.parse_args()

    # -------------------------------------------------------------- #

    # check usage of modality
    if args_.only_audio and args_.only_text:
        raise ValueError("Please check your usage of modalities.")

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
