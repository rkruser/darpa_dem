import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()

for j in range(10):
    for i in range(100):
        writer.add_scalar('Loss/train', np.random.random(), i)
        writer.add_scalar('Loss/test', np.random.random(), i)
    time.sleep(3)
