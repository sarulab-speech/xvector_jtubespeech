# x-vector extractor for Japanese speech
This repository provides a pre-trained model for extracting the [x-vector](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) (speaker representation vector). The model is trained using [JTubeSpeech corpus](https://github.com/sarulab-speech/jtubespeech), a Japanese speech corpus collected from YouTube.

このリポジトリは，[x-vector](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) (話者表現ベクトル) を抽出するための学習済みモデルを提供します．このモデルは，[JTubeSpeechコーパス](https://github.com/sarulab-speech/jtubespeech)と呼ばれる，YouTubeから収集した日本語音声から学習されています．

## Training configures / 学習時の設定
- The number of speakers: 1,233
- Sampling frequency: 16,000Hz
- Speaker recognition accuracy: 91% (test data) 
- Feature: 24-dimensional MFCC
- Dimensionality of x-vector: 512
- Other configurations: followed the ASV recipe for VoxCeleb in Kaldi.
  - In the opensourced model, model parameters of recognition layers following to the x-vector layer were randomized to protect data privacy.

## Installation
```bash
pip install xvector-jtubespeech
```

## Usage / 使い方
```
import numpy as np
from scipy.io import wavfile
import torch
from torchaudio.compliance import kaldi

from xvector_jtubespeech import XVector

def extract_xvector(
  model, # xvector model
  wav   # 16kHz mono
):
  # extract mfcc
  wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
  mfcc = kaldi.mfcc(wav, num_ceps=24, num_mel_bins=24) # [1, T, 24]
  mfcc = mfcc.unsqueeze(0)

  # extract xvector
  xvector = model.vectorize(mfcc) # (1, 512)
  xvector = xvector.to("cpu").detach().numpy().copy()[0]  

  return xvector

_, wav = wavfile.read("sample.wav") # 16kHz mono
model = XVector("xvector.pth")
xvector = extract_xvector(model, wav) # (512, )
```

## Contributors / 貢献者
- Takaki Hamada / 濱田 誉輝 (The University of Tokyo / 東京大学)
- Shinnosuke Takamichi / 高道 慎之介 (The University of Tokyo / 東京大学)

## License / ライセンス
MIT

## Others / その他
- The audio sample `sample.wav` was copied from [PJS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus).
