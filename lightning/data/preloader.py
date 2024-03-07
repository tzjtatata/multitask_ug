import os
from collections import Iterable
from typing import Callable


"""
    Thanks to @liuming, follow his multiprocess reader, we have a reader which preloads faster.
    Note that using preloader to load datas, need two function:
    1. imreader: is indepedent with dataset object, and return only bytes strings.
    2. decoder: can be a statistic method of dataset class, take bytes strings as input, \
       outputs standard Image object or np.Ndarray.
"""
def preloader(dataset, imreader: Callable, iterations: Iterable):
    from multiprocessing.dummy import Pool
    from tqdm import tqdm
    pool = Pool()  # use all threads by default
    result = []
    for r in tqdm(pool.imap(imreader, iter(iterations)), total=len(dataset)):
        result.append(r)
    pool.close()
    pool.join()

    return result
