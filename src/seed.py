import torch
import random
import numpy as np


# https://www.geeksforgeeks.org/deep-learning/reproducibility-in-pytorch/
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # https://docs.pytorch.org/docs/stable/generated/torch.mps.manual_seed.html
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True # section 3 https://www.geeksforgeeks.org/deep-learning/reproducibility-in-pytorch/
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
