# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:34:22 2019

@author: holmes
"""

import os 
import sys
from PIL import Image
import numpy as np


from inception_module import inception_module as incept_net

image_dir_parent_train = r"/path/to/datasets/plantclef2019"
image_dir_parent_test = r"/path/to/datasets/plantclef2019"
train_file = r"training_and_validation_list/clef2019_multilabel_train_256046.txt"
test_file = r"training_and_validation_list/clef2019_multilabel_test_256046.txt"

# model to restore
checkpoint_model = r"inception_v4.ckpt"
# save dir for checkpoint
checkpoint_save_dir = r"checkpoints_adam_multilabel"


batch = 64
input_size = (299,299,3)
numclasses = 10000
learning_rate = 0.0001
iterbatch = 4
max_iter = 500000
val_freq = 500
val_iter = 200

# construct and run the network
network = incept_net(
        batch = batch,
        iterbatch = iterbatch,
        numclasses = numclasses,
        input_size = input_size,
        image_dir_parent_train = image_dir_parent_train,
        image_dir_parent_test = image_dir_parent_test,
        train_file = train_file,
        test_file = test_file,
        checkpoint_model = checkpoint_model,
        save_dir = checkpoint_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter)


"""
LAYERS NAME
[ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
      'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
      'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
      'Mixed_7d','AuxLogits','Logits']

"""
