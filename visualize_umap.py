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
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
#import math
#import random
#import datetime
#import subprocess
#from collections import defaultdict, deque


import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP


def get_args_parser():
    #
    parser = argparse.ArgumentParser('tSNE', add_help=False)
    
    #----- General params -----#
    #parser.add_argument('--input_metadata', default='Attention_maps/van_run2/arrays/test_att.csv', type=str, help="""Path where input files are located""")
    parser.add_argument('--data_path', default='Attention_maps/van_run2/arrays/', type=str, help="""Path where input files are located""")
    #parser.add_argument('--att_head', default=0, type=int, help="""Num of the attention head to use for the tSNE. Default = 0""")
    parser.add_argument('--num_samples', default=-1, type=int, help="""Num of samples to plot t-SNE. Default = -1, to use all points in database""")
    parser.add_argument('--out_image', default = "umap.png", type=str, help="Name and path of output image. Default = \'umap.png\'")
    parser.add_argument('--out_results', default = "umap_results.npy", type=str, help="Name and path of output results. Default = \'umap_results.npy\'")
    parser.add_argument('--mode', default = "birads", type=str, choices=['birads', 'ACR', 'ID'], help="Mode to run UMAP Default = \'umap_results.npy\'")
    parser.add_argument('--subset', default = "test", type=str, choices=['train', 'test'], help="Data subset to perform UMAP. Default = \'test\'")
    parser.add_argument('--noise', default = 5, type=int, help="Level of noise. Default = 5")
    
    #----- tSNE params -----#
    parser.add_argument('--n_components', default = 2, type=int, help="Num of components for the tSNE. Default = 2")
    parser.add_argument('--n_neighbors', default = 15, type=int, help="Number of neighbors. Default = 15")
    parser.add_argument('--metric', default = "euclidean", type = str, choices=['euclidean', 'manhattan', "chebyshev", "minkowski", "cosine", "correlation"], help="Metric. Default = \'euclidean\'")
    #parser.add_argument('--n_iter', default = 300, type=int, help="Num of iterations. Default = 300")
    parser.add_argument('--verbose', default = 1, type=int, help="Verbose. Default = 1")
    return parser


def create_logger():
    #
    logging.raiseExceptions = False
    logger = logging.getLogger("this-logger")
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter) 
    logger.addHandler(consoleHandler)
    
    return logger


def Load_data (args): 
    #args
    input_meta = pd.read_csv (args.input_metadata)
    list_data = []
    
    if args.num_samples == -1: args.num_samples = len(input_meta)
    
    for file_ in input_meta["A2_d2"][:args.num_samples]:
        #
        data = np.load(args.data_path + file_)
        #list_mean.append(np.mean(data[args.att_head, :, :]))
        list_data.append(data[args.att_head, :, :].flatten())
    
    return np.asarray(list_data)

def Load_data_npy (args, return_meta = False): 
    #args
    
    if args.subset == "train": 
        data = np.load(args.data_path + "train_features.npy")
        input_meta = pd.read_csv (args.data_path + "train_features.csv")
    if args.subset == "test": 
        data = np.load(args.data_path + "test_features.npy")
        input_meta = pd.read_csv (args.data_path + "test_features.csv")
    
    if args.num_samples == -1: 
    #plt.title("t-SNE per " + args.mode + " att head: " + str(args.att_head) + "\n")
        #
        args.num_samples = len(data)
    else: 
        sampleInd = np.random.choice(data.shape[0], size=(args.num_samples,))
        data = data[sampleInd]
    
    if return_meta == True: 
        return np.asarray(data), input_meta
    else: 
        return np.asarray(data)


if __name__ == "__main__": 
    #
    parser = argparse.ArgumentParser('UMAP', parents=[get_args_parser()])
    args = parser.parse_args()
    
    #args = get_args_parser()
    logger = create_logger()
    #input_meta = pd.read_csv (args.input_metadata)
    
    time_start = time.time()
    logger.info("Start data loading.... This might take a while.")
    #data_subset = Load_data (args)
    data_subset, input_meta = Load_data_npy (args, return_meta = True); #print (data_subset.shape)
    logger.info("Data with shape " + str(data_subset.shape) + " successfully loaded! \n===================")
    
    logger.info("Starting UMAP with params: " + "\nThis might take a while....")
    
    ### Setup tSNE
    time_start = time.time()
    #tsne = tSNE(n_components = args.n_components, verbose = args.verbose, init='pca',
                #perplexity = args.perplexity, n_iter = args.n_iter, random_state=1)
    
    umap = UMAP(n_components = args.n_components, n_neighbors = args.n_neighbors, metric = args.metric, init='spectral', random_state=1)
    
    umap_results = umap.fit_transform(data_subset)
    logger.info("UMAP done! \n===================")
    
    time_start = time.time()
    np.save(args.out_results, umap_results)
    logger.info("Results saved! \n===================")
    
    #df_subset['tsne-2d-one'] = tsne_results[:,0]
    #df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    logger.info("Visualizing.... ")
    time_start = time.time()
    
    ''' Visualización de espacio incrustado '''
    noise = np. random. normal(0,args.noise,umap_results.shape[0])
    umap_results[:,0] = umap_results[:,0] + noise
    noise = np. random. normal(0,args.noise,umap_results.shape[0])
    umap_results[:,1] = umap_results[:,1] + noise
    
    df = pd.DataFrame(umap_results, columns = ['umap-one', 'umap-two'])
    df[args.mode] = input_meta[args.mode]
    cols = len(np.unique (input_meta[args.mode]))
    
    #plt.figure(figsize=(16,10))
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="umap-one", y="umap-two",
        hue=args.mode,
        palette=sns.color_palette("hls", n_colors = cols),
        data=df,
        legend="full",
        alpha=1
    )
    
    plt.title("UMAP per " + args.mode + "\n")
    plt.tight_layout()
    plt.savefig(args.out_image)
    plt.clf(); plt.close("all"); 
    
    logger.info("Figure saved as: " + str(args.out_image) + "\n===================\n")

