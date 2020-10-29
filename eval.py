import torch
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from dataset import LABEL_DICT


def evaluate(model, data_loader, device):
    loss = 0
    y_true, y_pred = [], []

    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss()
    iterator = tqdm(enumerate(data_loader), desc='eval_steps', total=len(data_loader))
    for step, batch in iterator:
        with torch.no_grad():

            # unpack and set inputs
            batch = map(lambda x: x.to(device), batch)
            audios, audio_masks, texts, labels = batch
            labels = labels.squeeze(-1).long()
            y_true += labels.tolist()

            # feed to model and get loss
            logit, hidden = model(audios, texts)
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
    acc = report['accuracy']
    loss /= len(data_loader)

    # logging
    log_template = "{}\tF1: {:.4f}\tPREC: {:.4f}\tREC: {:.4f}"
    logging.info((log_template + "\tACC: {:.4f}").format("TOTAL", f1, prec, rec, acc))
    for key, value in report.items():
        if key in LABEL_DICT:
            cur_f1 = value['f1-score']
            cur_prec = value['precision']
            cur_rec = value['recall']
            logging.info(log_template.format(key, cur_f1, cur_prec, cur_rec))
    logging.info('\n'+str(cm))
    return loss, f1


def main(args):
    return


if __name__ == "__main__":
    main(None)
