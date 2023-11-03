"""torch.hub configuration."""

dependencies = ["torch"]

import torch                                             # pylint: disable=wrong-import-position

from xvector_jtubespeech.network.xvector import _XVector # pylint: disable=wrong-import-position


URLS = {
    "xvector_jtubespeech": "https://raw.githubusercontent.com/sarulab-speech/xvector_jtubespeech/master/xvector.pth",
}


def xvector(progress: bool = True, pretrained: bool = True) -> _XVector:
    """
    `x-vector JTubeSpeech` utterance embedding model.

    Args:
        progress - Whether to show model checkpoint load progress
    """

    # Init
    model = _XVector(24, 1233)

    # Pretrained weights
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["xvector_jtubespeech"], map_location="cpu", progress=progress)
        model.load_state_dict(state_dict)

    # Mode
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model
