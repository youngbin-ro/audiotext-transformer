# korean-audiotext-transformer
> Multimodal Transformer for Korean Sentiment Analysis with Audio and Text Features

<br/>

## Overview
![overview](https://github.com/youngbin-ro/korean-audiotext-transformer/blob/master/images/overview.png?raw=true)

- **STEP1**: Convert input audio to text using [Google ASR API](https://cloud.google.com/speech-to-text/)
- **STEP2**: Extract MFCC feature from input audio
- **STEP3**: Conduct MLM on [KoBERT](http://aiopen.etri.re.kr/service_dataset.php) through colloquial review texts crawled from various services
- **STEP4**: Extract sentence embedding by using the text obtained in **STEP1** as an input variable of the BERT learned in **STEP3**
- **STEP5**: Obtain fused representation by using the MFCC feature (from **STEP2**) and the sentence embedding (from **STEP4**) as input variables of the Crossmodal Transformer ([Tsai et al., 2019](https://www.aclweb.org/anthology/P19-1656/))

<br/>

## Datasets
#### Korean Multimodal Sentiment Analysis Dataset

- 자율지능 디지털 동반자 [감정 분류용 데이터셋](http://aicompanion.or.kr/nanum/tech/data_introduce.php?offset=8&idx=23) (registration and authorization are needed for downloading)
- **Classes**: 분노(anger), 공포(fear), 중립(neutrality), 슬픔(sadness), 행복(happiness), 놀람(surprise), 혐오(disgust)
- **Provided Modality**: Video, Audio, Text
  - We only use audio and text data
  - When testing, text is obtained via ASR from audio without using the data provided
  - Vision modality is not considered in this project
- **Train / Dev / Test Split**
  
  - Based on audio: 8278 / 1014 / 1003
  - Based on text: 280 / 35 / 35
- **Preprocess**
  
  - Locate downloaded dataset as follows:
  
    ```
    korean-audiotext-transformer/    
    └── data/
        └── 4.1 감정분류용 데이터셋/
            ├── 000/
            ├── 001/
            ├── 002/
            ├── ...
            ├── 099/
            ├── participant_info.xlsx
            ├── rename_file.sh
            ├── Script.hwp
            └── test.py
    ```
  
  - Convert ```Script.hwp``` to ```script.txt```
  
    ```
    cd data/4.1 감정분류용 데이터셋
    hwp5txt Script.hwp --output script.txt
    ```
  
  - Generate ```{train, dev, test}.pkl```
  
    ```
    python preprocess.py \
      raw_path='./data/4.1 감정분류용 데이터셋' \
      script_path'./data/4.1 감정분류용 데이터셋/script.txt' \
      save_path='./data' \
      train_size=.8
    ```

- **Preprocessed Output** (<i>train.pkl</i>)

| person_ix |                                             audio |            Sentence | Emotion |
|----------:|--------------------------------------------------:|--------------------:|--------:|
|         0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         1 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         3 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         6 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         8 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |

<br/>

## Usage
#### Before training,

Download fine-tuned BERT, and locate the model as follows:

- The model is already fine-tuned with Korean sentimental analysis dataset
- We use this model as text embedding module (do not fine-tune anymore)

```
korean-audiotext-transformer/    
└── KoBERT/
    ├── args.bin
    ├── config.json
    ├── model.bin
    ├── tokenization.py
    └── vocab.list
```



#### Train AudioText Transformer

```shell
python train.py \
  --data_path='./data' \
  --bert_path='./KoBERT' \
  --save_path='./result' \
  --attn_dropout=.2 \
  --relu_dropout=.1 \
  --emb_dropout=.2 \
  --res_dropout=.1 \
  --out_dropout=.1 \
  --n_layers=4 \
  --d_model=64 \
  --n_heads=8 \
  --lr=1e-5 \
  --epochs=10 \
  --batch_size=16 \
  --n_mfcc=40
```






























