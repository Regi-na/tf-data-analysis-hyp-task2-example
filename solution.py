import pandas as pd
import numpy as np
from hyppo.ksample import Energy, MMD

chat_id = 947352272 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    return MMD(compute_kernel="laplacian", gamma=0.55).test(x, y)[1] < 0.06

