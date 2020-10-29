import argparse
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from dataset import LABEL_DICT, get_data_loader
from model import MultimodalTransformer


def evaluate(model,
             data_loader,
             device):
    loss = 0
    y_true, y_pred = [], []

    model.eval()
    model.zero_grad()
    loss_fct = torch.nn.CrossEntropyLoss()
    iterator = tqdm(enumerate(data_loader), desc='eval_steps', total=len(data_loader))
    for step, batch in iterator:
        with torch.no_grad():

            # unpack and set inputs
            batch = map(lambda x: x.to(device) if x is not None else x, batch)
            audios, a_mask, texts, t_mask, labels = batch
            labels = labels.squeeze(-1).long()
            y_true += labels.tolist()

            # feed to model and get loss
            logit, hidden = model(audios, texts, a_mask, t_mask)
            cur_loss = loss_fct(logit, labels.view(-1))
            loss += cur_loss.item()
            y_pred += logit.max(dim=1)[1].tolist()

    # evaluate with metrics
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABEL_DICT))),
        target_names=list(LABEL_DICT.keys()),
        output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    f1 = report['macro avg']['f1-score']
    prec = report['macro avg']['precision']
    rec = report['macro avg']['recall']
    loss /= len(data_loader)

    # logging
    log_template = "{}\tF1: {:.4f}\tPREC: {:.4f}\tREC: {:.4f}"
    logging.info(log_template.format("TOTAL", f1, prec, rec))
    for key, value in report.items():
        if key in LABEL_DICT:
            cur_f1 = value['f1-score']
            cur_prec = value['precision']
            cur_rec = value['recall']
            logging.info(log_template.format(key, cur_f1, cur_prec, cur_rec))
    logging.info('\n'+str(cm))
    return loss, f1


def main(args):
    data_loader = get_data_loader(
        args=args,
        data_path=args.data_path,
        bert_path=args.bert_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        split=args.split
    )

    model = MultimodalTransformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_classes=args.n_classes,
        only_audio=args.only_audio,
        only_text=args.only_text,
        d_audio_orig=args.n_mfcc,
        d_text_orig=768,  # BERT hidden size
        d_model=args.d_model,
        attn_mask=args.attn_mask
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path))

    # evaluation
    logging.info('evaluation starts')
    model.zero_grad()
    evaluate(model, data_loader, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--only_audio', action='store_true')
    parser.add_argument('--only_text', action='store_true')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--bert_path', type=str, default='./KoBERT')
    parser.add_argument('--model_path', type=str, default='./result/epoch2-loss1.5351-f10.3848.bin')
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)

    # architecture
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--attn_mask', action='store_false')

    # data processing
    parser.add_argument('--max_len_audio', type=int, default=400)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--resample_rate', type=int, default=16000)
    parser.add_argument('--n_fft_size', type=int, default=400)
    parser.add_argument('--n_mfcc', type=int, default=40)

    args_ = parser.parse_args()

    # -------------------------------------------------------------- #
    
    # check usage of modality
    if args_.only_audio and args_.only_text:
        raise ValueError("Please check your usage of modalities.")

    # seed and device setting
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args_.device = device_

    # log setting
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    main(args_)
