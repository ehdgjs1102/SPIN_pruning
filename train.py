from utils import TrainOptions
from train import Trainer

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
