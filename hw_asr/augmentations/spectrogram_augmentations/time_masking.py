import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, time_mask_param, p):
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
