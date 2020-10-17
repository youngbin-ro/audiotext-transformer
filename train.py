import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--do_audio', action='store_true')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--bert_path', type=str, default='./KoBERT')

    # dropouts
    parser.add_argument('--attn_dropout', type=float, default=.2)
    parser.add_argument('--relu_dropout', type=float, default=.1)
    parser.add_argument('--emb_dropout', type=float, default=.2)
    parser.add_argument('--res_dropout', type=float, default=.1)
    parser.add_argument('--out_dropout', type=float, default=.1)



