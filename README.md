# X-Vectors in JTubeSpeech Corpus

[X-Vectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) をJTubeSpeechコーパスで訓練したモデルです。

## usage

```
>>> from xvector_jtubespeech import XVector
>>> model = XVector()
>>> xvector = model.vectorize(mfcc) # mfcc: num_mel_bins = 24
```

`mfcc` は [`torchaudio.compliance.kaldi.mfcc`](https://pytorch.org/audio/stable/compliance.kaldi.html#mfcc) を用いて、
`num_mel_bins = 24` で計算することを想定しています。
