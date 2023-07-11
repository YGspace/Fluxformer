import torch
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from data import build_train_dataset
from gmflow.gmflow import GMFlow
from loss import flow_loss_func
from evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                      create_sintel_submission, create_kitti_submission, inference_on_dir)

from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed

from ptflops import get_model_complexity_info

from thop import profile
from thop import clever_format
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.cuda.device(0):
  net = model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=4,
                   ).to(device)
  # macs, params = get_model_complexity_info(net, (2, 3, 224, 224), as_strings=True,
  #                                          print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

  img1 = torch.randn(1, 3, 224, 224).to(device)
  img2 = torch.randn(1, 3, 224, 224).to(device)

  mac, params = profile(model, inputs=(img1, img2))
  mac, params = clever_format([mac, params], "%.3f")
  print(mac)
  print(params)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', mac))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

