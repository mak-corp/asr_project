from torch.utils.data import Sampler, BatchSampler, SequentialSampler, RandomSampler


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=True):
        super().__init__(data_source)
        self.epochs = 0
        self.sequential = BatchSampler(SequentialSampler(data_source), batch_size=batch_size, drop_last=drop_last)
        self.random = BatchSampler(RandomSampler(data_source), batch_size=batch_size, drop_last=drop_last)
        self.len = len(data_source)

    def __iter__(self):
        self.epochs += 1
        if self.epochs == 1:
            yield from self.sequential
        else:
            yield from self.random

    def __len__(self):
        return self.len
