import itertools
import time

import torch
import math
from collections.abc import Sized
from torch.utils.data import Sampler
from typing import List


class MultipleSampler(Sampler[List[int]]):

    def __init__(self,
                 data_source: Sized,
                 batch_size: int,
                 drop_last: bool,
                 num_tasks: int) -> None:
        if isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_tasks = num_tasks
        print("Now Using Multiple Sampler with {} times repeat.".format(self.num_tasks))

    def __iter__(self):
        n = len(self.data_source)
        generator = torch.Generator()
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)

        # By Default, we use drop last setting.
        real_n = (n // self.batch_size) * self.batch_size
        random_sample = torch.randperm(n, generator=generator).tolist()[:real_n]

        length = len(self.data_source) // self.batch_size
        for _ in range(length):
            current = random_sample[:self.batch_size]
            random_sample = random_sample[:self.batch_size:]
            for _ in range(self.num_tasks):
                yield current

    def __len__(self):
        return (len(self.data_source) // self.batch_size) * self.num_tasks


def check_valid(batch):
    assert len(batch) == len(set(batch)), print(batch, len(batch), len(set(batch)))
