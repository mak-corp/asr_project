import librosa
from torch import Tensor
import torch

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, data: Tensor):
        return torch.from_numpy(librosa.effects.time_stretch(data.cpu().numpy(), rate=self.rate))
