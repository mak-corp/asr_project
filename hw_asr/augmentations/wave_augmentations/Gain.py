import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, gain, gain_type='amplitude'):
        self._aug = torchaudio.transforms.Vol(gain, gain_type)

    def __call__(self, data: Tensor):
        return self._aug(data)
