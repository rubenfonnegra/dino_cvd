# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import pickle
import argparse

import torch
#from torch import nn
import torch.distributed as dist
#import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
#from torchvision import transforms as pth_transforms
#from torchvision import datasets
from PIL import Image, ImageFile
import numpy as np
import pandas as pd

import utils
import vision_transformer as vits
#from eval_knn import extract_features


class dino_args(): 
    def __init__(self):
        #
        self.data_path = '/path/to/revisited_paris_oxford/'
        self.dataset = 'none'
        self.multiscale = False
        self.imsize = 480
        self.pretrained_weights = "/home/ruben-kubuntu/Devs/dino/Results/tiny_run0/checkpoint0300.pth"
        self.train_data_path = "/home/ruben-kubuntu/Devs/dino/Data/breast_data/d2/dino_tr/"
        self.test_data_path = "/home/ruben-kubuntu/Devs/dino/Data/breast_data/d2/dino_ts/"
        self.output_dir = "/home/ruben-kubuntu/Devs/dino/Features/tiny/"
        self.use_cuda = True
        self.arch = "vit_tiny"
        self.patch_size = 16
        self.checkpoint_key = "teacher"
        self.num_workers = 10
        self.dist_url = "env://"
        self.local_rank = 0
        self.batch_size = 1
        

def build_dino (args): 
    #
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    if args.use_cuda:
        model.cuda()
    model.eval()
    
    # load pretrained weights
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.train_names` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    elif args.arch == "vit_small" and args.patch_size == 16:
        print("Since no pretrained weights have been provided, we load pretrained DINO weights on Google Landmark v2.")
        model.load_state_dict(torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))
    else:
        print("Warning: We use random weights.")
    
    return model


def extract_dino_features (args, model, samples):
    #
    features = None
    index = torch.from_numpy(np.zeros((args.batch_size)))
    samples = torch.stack(list(samples), dim=0)
    samples = torch.squeeze(samples)
    
    samples = samples.cuda(non_blocking=True)
    index = index.cuda(non_blocking=True)
    
    if args.multiscale:
        feats = utils.multi_scale(samples, model)
    else:
        feats = model(samples).clone()
    
    return feats
