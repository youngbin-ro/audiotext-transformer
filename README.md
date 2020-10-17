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
#### 1. Korean Multimodal Sentiment Analysis Dataset

- 자율지능 디지털 동반자 [감정 분류용 데이터셋](http://aicompanion.or.kr/nanum/tech/data_introduce.php?offset=8&idx=23) (registration and authorization are needed for downloading)
- **Classes**: 분노(anger), 공포(fear), 중립(neutrality), 슬픔(sadness), 행복(happiness), 놀람(surprise), 혐오(disgust)
- **Provided Modality**: Video, Audio, Text
  - We only use audio data
  - Text feature is obtained via ASR from audio without using the data provided
  - Vision modality is not considered in this project
- **Train / Dev / Test Split**
  - Based on audio: 8278 / 1014 / 1003
  - Based on text: 280 / 35 / 35
- **Example**
| person_ix |                                             audio |            Sentence | Emotion |
|----------:|--------------------------------------------------:|--------------------:|--------:|
|         0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         1 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         3 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         6 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |
|         8 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... | 오늘 입고 나가야지. |    행복 |



#### 2. Korean Service Review Texts

> The dataset for STEP3 (KoBERT further pre-training)

- **Download**: [Train](https://drive.google.com/file/d/1GvgWPKXVATj86GjZEJ4Xzzf-TiiOmKL3/view?usp=sharing) / [Dev](https://drive.google.com/file/d/1_O5_2xouFl3GYijONxqmPBqsP_dJM4QR/view?usp=sharing) / [Test](https://drive.google.com/file/d/1l5lfoiwZ-SAe0coK0kfVDjRK6EqMlVcT/view?usp=sharing)
- **Classes**: 긍정(positive), 부정(negative)
- **Domains**: 어플리케이션(application), 화장품(cosmetic), 영화(movie)
- **Information**
  - Crawled from [Google Play](https://play.google.com/store?hl=ko), [Naver Movie](https://movie.naver.com/), [Glowpick](https://www.glowpick.com/)
  - We do not use class and domain information for training (only use sentence itself for unsupervised masked language modeling)
  - The code for MLM is not provided in this repository (please refer [here](https://github.com/JhnLee/pytorch-bert-korean))
- **Example**
|                                                     Sentence | Emotion |      Domain |
| -----------------------------------------------------------: | ------: | ----------: |
|        굿굿, 사용 후 탄력이나 피부결 느낌 좋은데 가격이 ㄷㄷ |    긍정 |    cosmetic |
|              광고에게 사기당한 영화. 귀신이 안나오기는 개뿔. |    부정 |       movie |
|                                           내 시간을 돌려줘.. |    부정 |       movie |
| 10살 나의 생일 날, 둘 째 언니가 물었다. 올 해 생일선물로는 ... |    긍정 |       movie |
|                                          왜 설치가 안되냐고; |    부정 | application |









































