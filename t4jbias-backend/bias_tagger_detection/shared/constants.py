"""
just a lil global variable so everybody knows whether
we have a gpu or not
"""

import torch

# automatically set CUDA, set this to False to disable GPU
CUDA = (torch.cuda.device_count() > 0)

# in case cuDNN is causing issues, it can be disabled
# torch.backends.cudnn.enabled = False

