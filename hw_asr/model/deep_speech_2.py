from hw_asr.base import BaseModel
import torch
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.stride = stride

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True),
        ])

    def zero_paddings(self, x, lengths):
        mask = torch.zeros_like(x, dtype=torch.bool)
        time = x.shape[-1]
        for i, length in enumerate(lengths):
            if time > length:
                mask[i].narrow(dim=-1, start=length, length=time-length).fill_(True)
        x.masked_fill(mask, 0)

        return x

    def forward(self, inputs):
        '''
        :param x: input, shape BxCxFxT
        :param lengths: real input lengths, shape: B
        '''
        x, lengths = inputs
        lengths = lengths // self.stride[1]

        for layer in self.layers:
            x = self.zero_paddings(layer(x), lengths)

        return x, lengths


class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=True, bidirectional=True):
        super(RNNBlock, self).__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=bidirectional, bias=True)
        self.batch_norm = nn.BatchNorm1d(num_features=input_size) if batch_norm else None


    def forward(self, inputs):
        '''
        :param x: input, shape TxBx*
        :param lengths: real input lengths, shape: B
        '''
        x, lengths = inputs

        time, batch, hidden = x.shape
        if self.batch_norm is not None:
            x = self.batch_norm(x.view(time * batch, hidden)).view(time, batch, hidden)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)

        if self.bidirectional:
            hidden = x.shape[-1] // 2
            x = x[:,:,:hidden] + x[:,:,hidden:]

        return x, lengths


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, hidden_rnn_size, n_rnns, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.conv = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
        )
        rnn_input_size = n_feats // 4 * 32

        self.rnn = nn.Sequential(
            RNNBlock(rnn_input_size, hidden_rnn_size, batch_norm=False, bidirectional=True),
            *[
                RNNBlock(hidden_rnn_size, hidden_rnn_size, batch_norm=True, bidirectional=True)
                for _ in range(1, n_rnns)
            ],
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_rnn_size),
            nn.Linear(hidden_rnn_size, n_class, bias=False)
        )

    def forward(self, spectrogram, **batch):
        x, lengths = self.conv((spectrogram.unsqueeze(1), batch['spectrogram_length']))

        batch, ch, freq, time = x.shape
        x = x.view(batch, ch * freq, time).permute(2, 0, 1).contiguous()
        x, lengths = self.rnn((x, lengths))

        time, batch, hidden = x.shape
        x = self.fc(x.view(time * batch, hidden))
        x = x.view(time, batch, -1).permute(1, 0, 2).contiguous()

        return {
            "logits": x
        }

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2