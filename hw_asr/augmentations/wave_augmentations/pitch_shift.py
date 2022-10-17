import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, sample_rate, n_steps):
        self._aug = torchaudio.transforms.PitchShift(sample_rate, n_steps)

    def __call__(self, data: Tensor):
        return self._aug(data)
