# X-Vectors in JTubeSpeech Corpus

[X-Vectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) を [JTubeSpeechコーパス](https://github.com/sarulab-speech/jtubespeech) で訓練したモデルです。

音声から抽出したMFCCを固定長の話者表現に変換する機能を提供しています。
MFCCは [`torchaudio.compliance.kaldi.mfcc`](https://pytorch.org/audio/stable/compliance.kaldi.html#mfcc) を用いて、
`num_ceps=24, num_mel_bins = 24` の設定で計算することを想定しています。

## data

### 音声ファイル

* 拡張子： `.wav`
* サンプリング周波数：16000Hz

### コーパス

* 使用話者数：1233人

### 認識性能

...

## usage

```
>>> import numpy as np
>>> from scipy.io import wavfile
>>> import torch
>>> from torchaudio.compliance import kaldi

>>> fs, w = wavfile.read("/path/to/speech.wav")
>>> w = w.astype(np.float32)
>>> w = torch.from_numpy(w)
>>> w = w.unsqueeze(0)
>>> mfcc = kaldi.mfcc(w, num_ceps=24, num_mel_bins=24)
>>> mfcc = mfcc.unsqueeze(0) # mfcc.shape: (1, T, 24)

>>> from xvector_jtubespeech import XVector
>>> model = XVector()
>>> xvector = model.vectorize(mfcc)
>>> xvector.shape
torch.Size([1, 512])
```
