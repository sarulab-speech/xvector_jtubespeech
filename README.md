# X-Vectors in JTubeSpeech Corpus

[X-Vectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) をJTubeSpeechコーパスで訓練したモデルです。

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
>>> xvector = model.vectorize(mfcc) # mfcc: num_mel_bins = 24
>>> xvector.shape
torch.Size([1, 512])
```

`mfcc` は [`torchaudio.compliance.kaldi.mfcc`](https://pytorch.org/audio/stable/compliance.kaldi.html#mfcc) を用いて、
`num_ceps=24, num_mel_bins = 24` で計算することを想定しています。
