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
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from torchvision import datasets
from PIL import Image, ImageFile
import numpy as np
import pandas as pd

import utils
import vision_transformer as vits


def extract_dino_features(model, data_loader, use_cuda=True, multiscale=False, return_csv=False, save_path = "Features/"):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = []
    i = 0; all_samples = []; all_labels = []; 
    for samples, index in metric_logger.log_every(data_loader, 10):
        if use_cuda: 
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
        
        #-----
        sample_fname, sample_class = data_loader.dataset.samples[i]; i += 1
        pos_fname = str.rfind(sample_fname, "/")
        sample_fname = sample_fname[pos_fname+1:] 
        all_samples.append(sample_fname)
        all_labels.append(sample_class)
        #print (sample_fname) #csv_name
        #-----
        
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()
        
        if not features:
            print(f"Storing features into tensor of shape [{len(data_loader.dataset)}, {feats.shape[-1]}]")
        
        if use_cuda: 
            features.extend(feats.cpu().detach().numpy().reshape([1,-1]))
        else: 
            features.expand(feats.detach().numpy())
        
    if return_csv == True: 
        return np.array(features), all_samples, all_labels
    else:
        return np.array(features)


def main(): 
    parser = argparse.ArgumentParser('Image Retrieval on revisited Paris and Oxford')
    parser.add_argument('--data_path', default='/path/to/revisited_paris_oxford/', type=str)
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--train_data_path', default='', type=str, help="Path to train samples.")
    parser.add_argument('--test_data_path', default=None, type=str, help="Path to test samples.")
    parser.add_argument('--output_dir', default='', type=str, help="Path to test samples.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--gpu', default=0, type=int, help='GPU to use')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--num-channels", default=3, type=int, help="""Number of input channels""")
    parser.add_argument("--port", default='29500', type=str, help="""port for parallelization""")
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    if args.num_channels == 3: 
        transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Resize(args.imsize, pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #pth_transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else: 
        transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Resize(args.imsize, pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            pth_transforms.Grayscale(1),
        ])
    
    dataset_train = datasets.ImageFolder(args.train_data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    if args.test_data_path: 
        dataset_query = datasets.ImageFolder(args.test_data_path, transform=transform)
    
    if args.test_data_path: 
        data_loader_query = torch.utils.data.DataLoader(
            dataset_query,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    print(f"train: {len(dataset_train)} imgs") 
    if args.test_data_path: 
        print(f"query: {len(dataset_query)} imgs")
    
    
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_channels=args.num_channels, num_classes=0)
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

    ############################################################################
    
    os.makedirs(args.output_dir, exist_ok = True)
    
    # Step 1: extract features
    train_features, train_names, train_labels = extract_dino_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale, return_csv = True, save_path = args.output_dir + "train/")
    #train_features = nn.functional.normalize(train_features, dim=1, p=2)
    
    csv_dict = {"namefile": train_names, "label": train_labels}
    csv_dict = pd.DataFrame.from_dict(csv_dict)
    csv_dict.to_csv(args.output_dir + "train_features.csv", index = False)
    np.save (args.output_dir + "train_features.npy", train_features)
    
    
    if args.test_data_path: 
        query_features, query_names, query_labels = extract_dino_features(model, data_loader_query, args.use_cuda, multiscale=args.multiscale, return_csv = True, save_path = args.output_dir + "test/")
        #query_features = nn.functional.normalize(query_features, dim=1, p=2)
        np.save (args.output_dir + "test_features.npy", query_features)
        
        csv_dict = {"namefile": query_names, "label": query_labels}
        csv_dict = pd.DataFrame.from_dict(csv_dict)
        csv_dict.to_csv(args.output_dir + "test_features.csv", index = False)
    

if __name__ == '__main__':
    """
    param = sys.argv.append
    param ("--arch"); param("vit_small"); param ("--imsize"); param("256"); 
    param ("--gpu"); param("5"); param ("--multiscale"); param("0"); 
    param ("--train_data_path"); param ("Data/he_data/he_7k/"); 
    #param ("--test_data_path"); param("Data/he_data/CRC-VAL-HE-7K-CONT/"); 
    param ("--pretrained_weights"); param("/scr/rfonnegr/sources/pretrains/dino/checkpoint.pth") #param("/scr/rfonnegr/sources/pretrains/dino/dino_cells.pth");
    param ("--output_dir"); param("Features/he_7k_jc/"); 
    param ("--num-channels"); param("3"); 
    """
    main()