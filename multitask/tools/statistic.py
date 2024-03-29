from typing import List
import numpy as np


def cal_trust_interval(data: List[float]):
    m = np.mean(data)
    s = np.std(data)
    width = 1.96 * (s / np.sqrt(len(data) * 1.0))
    return m, width

