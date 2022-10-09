import logging
from typing import List
from collections import defaultdict

from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = defaultdict(list)
    for dataset_item in dataset_items:
        for k, v in dataset_item.items():
            result_batch[k].append(v)

    result_batch["spectrogram_length"] = torch.tensor([spectrogram.shape[2] for spectrogram in result_batch["spectrogram"]])
    spectrograms = [spectrogram.squeeze().permute(1, 0) for spectrogram in result_batch["spectrogram"]]
    result_batch["spectrogram"] = pad_sequence(spectrograms, batch_first=True).permute(0, 2, 1)

    result_batch["text_encoded_length"] = torch.tensor([text_encoded.shape[1] for text_encoded in result_batch["text_encoded"]])
    result_batch["text_encoded"] = pad_sequence([text_encoded.squeeze() for text_encoded in result_batch["text_encoded"]], batch_first=True)
    
    return dict(result_batch)