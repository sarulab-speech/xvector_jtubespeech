from pathlib import Path
import torch
import torch.nn as nn
from .tdnn import TDNN
# from xvector_jtubespeech.network.tdnn import TDNN

def XVector(model_path="./xvector.pth"):
    model_not_exist_msg = (
        f"[error] dumped file of model's state dict does not exist at {model_path}"
    )
    assert Path(model_path).exists(), model_not_exist_msg

    model = _XVector(24, 1233)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


class _XVector(nn.Module):
    def __init__(self, in_dim, classes, stat_dim=1500, hidden_dim=512):
        super(_XVector, self).__init__()

        self.stat_dim = stat_dim
        self.hidden_dim = hidden_dim

        # from Table 1. of the X-Vectors paper:
        # https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        self.frames = nn.Sequential(
            TDNN(5, 1, in_dim, hidden_dim),
            TDNN(3, 2, hidden_dim, hidden_dim),
            TDNN(3, 3, hidden_dim, hidden_dim),
            TDNN(1, 1, hidden_dim, hidden_dim),
            TDNN(1, 1, hidden_dim, stat_dim),
        )
        self.segment_6 = nn.Linear(stat_dim * 2, hidden_dim)
        self.segment_7 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, classes)

    def vectorize(self, x):
        x = self.frames(x)

        # stats-pooling
        mean = torch.mean(x, 1)
        std = torch.std(x, 1)
        x = torch.cat((mean, std), 1)

        vec = self.segment_6(x)

        return vec
