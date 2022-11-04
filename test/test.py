import numpy as np
from scipy.io import wavfile
import torch
from torchaudio.compliance import kaldi
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
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
  xvector = model.vectorize(mfcc) # [1, 512]
  xvector = xvector.to("cpu").detach().numpy().copy()[0]  

  return xvector

_, wav = wavfile.read("sample.wav") # 16bit mono
model = XVector("xvector.pth")
xvector = extract_xvector(model, wav) # [512, ]
print(xvector.shape)