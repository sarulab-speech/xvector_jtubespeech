import os
import numpy as np
import torchaudio
from torchaudio.compliance import kaldi

from .xvector_jtubespeech import XVector as XVectorImpl

class XVector:
    def __init__(self):
        model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "xvector.pth"
            )
        self.model = XVectorImpl(model_path)

    def calc_from_wav(self, w):
        w *= np.iinfo(np.int16).max
        w = w.unsqueeze(0)
        mfcc = kaldi.mfcc(w, num_ceps=24, num_mel_bins=24)
        mfcc = mfcc.unsqueeze(0) # mfcc.shape: (1, T, 24)

        return self.model.vectorize(mfcc)

    def __call__(self, wav_fname):
        data = self.load_wav(wav_fname)
        return self.calc_from_wav(data)[0]

    def load_wav(self, wav_path, sample_rate=16000):
        waveform, sr = torchaudio.load(wav_path, normalize=True)
        if sample_rate != sr:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        return waveform[0]

