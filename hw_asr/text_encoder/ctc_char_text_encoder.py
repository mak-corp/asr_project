from typing import List, Union, NamedTuple

import torch
from operator import itemgetter
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    EMPTY_IND = 0

    def __init__(self, alphabet: Union[List[str], str] = None,
                 use_custom_beam_search = True,
                 vocab_path = None, lm_path = None,
                 alpha=0.7, beta=0.1):
        if isinstance(alphabet, str):
            alphabet = [ch for ch in alphabet]
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        if not use_custom_beam_search and vocab_path and lm_path:
            print("Use fast CTC beam search with LM")
            with open(vocab_path) as f:
                unigrams = [line.strip() for line in f]

            up_vocab = [ch.upper() for ch in [''] + self.alphabet]
            self.decoder = build_ctcdecoder(
                up_vocab,
                alpha=alpha,
                beta=beta,
                kenlm_model_path=lm_path,
                unigrams=unigrams,
            )
        else:
            print("Use custom CTC beam search")
            self.decoder = None

    def ctc_decode(self, inds: List[int]) -> str:
        last_ind = self.EMPTY_IND
        decoded_inds = []
        for ind in inds:
            if ind != last_ind:
                last_ind = ind
                if ind != self.EMPTY_IND:
                    decoded_inds.append(self.ind2char[ind])

        return ''.join(decoded_inds)

    def _extend_and_merge(self, token_probs, dp):
        new_dp = defaultdict(float)
        for (text, last_char), proba in dp:
            for ind, token_proba in enumerate(token_probs):
                new_char = self.ind2char[ind]
                if new_char != last_char and last_char != self.EMPTY_TOK:
                    new_text = text + last_char
                else:
                    new_text = text
                new_dp[(new_text, new_char)] += proba * token_proba
        return new_dp

    def _cut_beams(self, dp, beam_size):
        return sorted(dp.items(), key=itemgetter(1), reverse=True)[:beam_size]

    def custom_ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        dp = [(('', self.EMPTY_TOK), 1.0)]
        for token_probs in probs[:probs_length]:
            dp = self._extend_and_merge(token_probs, dp)
            dp = self._cut_beams(dp, beam_size)
        dp = self._extend_and_merge((1,0), dp)
        dp = self._cut_beams(dp, beam_size)

        return [Hypothesis(text.strip(), proba) for (text, last_char), proba in dp]

    def fast_ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        logits = probs[:probs_length].cpu().numpy()
        text = self.decoder.decode(logits, beam_width=beam_size).lower()
        return [Hypothesis(text, 1.0)]

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        if self.decoder:
            return self.fast_ctc_beam_search(probs, probs_length, beam_size)
        return self.custom_ctc_beam_search(probs, probs_length, beam_size)
