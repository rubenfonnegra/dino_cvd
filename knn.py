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
import argparse
import numpy as  np
import pandas as pd

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


@torch.no_grad()
def load_dino_features (args, return_names = False): 
    #args
    train_features = np.load(args.data_path + "train_features.npy")
    test_features = np.load(args.data_path + "test_features.npy")
    train_names = pd.read_csv (args.data_path + "train_features.csv")
    test_names = pd.read_csv (args.data_path + "test_features.csv")
    train_names = np.array(train_names["label"])
    test_names = np.array(test_names["label"])

    if args.num_samples == -1: 
        args.num_samples = len(test_names)
    else: 
        sampleInd = np.random.choice(len(test_names), size=(args.num_samples,))
        train_features = train_features[sampleInd]
        test_features = test_features[sampleInd]
        train_names = train_names[sampleInd]
        test_names = test_names[sampleInd]
    
    if return_names == True: 
        return train_features, test_features, train_names, test_names
    else: 
        return train_features, test_features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    #print (num_classes)
    top1, top3, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top3 = top3 + correct.narrow(1, 0, min(3, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    #print (predictions.size(), correct.size()); print (targets.size())
    top1 = top1 * 100.0 / total
    top3 = top3 * 100.0 / total
    return top1, top3


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def main(): 
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--num_channels', default=3, type=int, help='Num of image channels')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    #parser.add_argument('--dump_features', default=None,
        #help='Path where to save computed features, empty for no saving')
    #parser.add_argument('--load_features', default=None, help="""If the features have
        #already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_samples', default=-1, type=int, help="""Num of samples to plot t-SNE. Default = -1, to use all points in database""")
    
    # GPU options
    parser.add_argument('--gpu', default=0, type=int, help='GPUs to be used (default: %(default)s)')
    parser.add_argument("--port", default='29500', type=str, help="""port for parallelization""")

    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    
    print ("Loading features.....")
    train_features, test_features, train_labels, test_labels = load_dino_features(args = args, return_names = True)
    train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top3 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, args.temperature, num_classes=4)
            print(f"{k}-NN classifier result: Top1: {top1}, Top3: {top3}")
    dist.barrier()


if __name__ == '__main__':
    """
    param = sys.argv.append
    param ("--arch"); param("vit_small"); param ("--patch_size"); param("16"); 
    param ("--gpu"); param("0"); param ("--checkpoint_key"); param("teacher"); 
    param ("--pretrained_weights"); param("Results/sm_cvd_3/checkpoint.pth")
    param ("--data_path"); param ("/home/ruben-kubuntu/Devs/dino_cvd/Features/sm_cvd_3/")
    param ("--num_channels"); param("1")
    """
    main()
