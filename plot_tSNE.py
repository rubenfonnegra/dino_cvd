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
from sklearn.manifold import TSNE as tSNE


def get_args_parser():
    #
    parser = argparse.ArgumentParser('tSNE', add_help=False)
    
    #----- General params -----#
    #parser.add_argument('--input_metadata', default='Attention_maps/van_run2/arrays/test_att.csv', type=str, help="""Path where input files are located""")
    parser.add_argument('--data_path', default='Attention_maps/van_run2/arrays/', type=str, help="""Path where input files are located""")
    #parser.add_argument('--att_head', default=0, type=int, help="""Num of the attention head to use for the tSNE. Default = 0""")
    parser.add_argument('--num_samples', default=-1, type=int, help="""Num of samples to plot t-SNE. Default = -1, to use all points in database""")
    parser.add_argument('--out_image', default = "umap.png", type=str, help="Name and path of output image. Default = \'umap.png\'")
    parser.add_argument('--out_results', default = None, type=str, help="Name and path of output results. Default = \'umap_results.npy\'")
    #parser.add_argument('--mode', default = "birads", type=str, choices=['birads', 'ACR', 'ID'], help="Mode to run UMAP Default = \'umap_results.npy\'")
    parser.add_argument('--subset', default = "both", type=str, choices=['both', 'reals', "fakes"], help="Data subset to perform UMAP. Default = \'test\'")
    parser.add_argument('--noise', default = 5, type=int, help="Level of noise. Default = 5")
    
    #----- tSNE params -----#
    parser.add_argument('--n_components', default = 2, type=int, help="Num of components for the tSNE. Default = 2")
    parser.add_argument('--perplexity', default = 40, type=int, help="Perplexity value. Default = 40")
    parser.add_argument('--n_iter', default = 300, type=int, help="Num of iterations. Default = 300")
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


def load_data_npy (args, return_names = False): 
    #args
    
    if args.subset == "reals": 
        features = np.load(args.data_path + "real_features.npy")
        names = pd.read_csv (args.data_path + "real_features.csv")
        targets = np.ones([len(features), 1])
    if args.subset == "fakes": 
        features = np.load(args.data_path + "fake_features.npy")
        names = pd.read_csv (args.data_path + "fake_features.csv")
        targets = np.zeros([len(features), 1])
    if args.subset == "both": 
        real_features = np.load(args.data_path + "real_features.npy")
        fake_features = np.load(args.data_path + "fake_features.npy")
        real_names = pd.read_csv (args.data_path + "real_features.csv")
        fake_names = pd.read_csv (args.data_path + "fake_features.csv")
        real_targets = np.ones([len(real_features), 1])
        fake_targets = np.zeros([len(fake_features), 1])
        features = np.concatenate([real_features, fake_features], axis = 0)
        names = np.concatenate([real_names, fake_names], axis = 0)
        targets = np.concatenate([real_targets, fake_targets], axis = 0)
    
    if args.num_samples == -1: 
        args.num_samples = len(names)
    else: 
        sampleInd = np.random.choice(len(names), size=(args.num_samples,))
        features = features[sampleInd]
        names = names[sampleInd]
        targets = targets[sampleInd]
    
    if return_names == True: 
        return features, targets, names
    else: 
        return features, targets


def main(): 
    #
    parser = argparse.ArgumentParser('tSNE', parents=[get_args_parser()])
    args = parser.parse_args()
    logger = create_logger()
    
    time_start = time.time()
    logger.info("Start data loading.... This might take a while.")
    #data_subset = Load_data (args)
    features, targets, names = load_data_npy (args, return_names = True); #print (data_subset.shape)
    logger.info("Data with shape " + str(features.shape) + " successfully loaded! \n===================")
    
    
    logger.info("Starting t-SNE with params: " + "\nThis might take a while....")
    
    ### Setup tSNE
    time_start = time.time()
    tsne = tSNE(n_components = args.n_components, verbose = args.verbose, init='pca',
                perplexity = args.perplexity, n_iter = args.n_iter, random_state=40)
    
    tsne_results = tsne.fit_transform(features)
    logger.info("t-SNE done! \n===================")
    
    time_start = time.time()
    if args.out_results: 
        np.save(args.out_results, tsne_results)
    logger.info("Results saved! \n===================")
    
    #df_subset['tsne-2d-one'] = tsne_results[:,0]
    #df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    logger.info("Visualizing.... ")
    time_start = time.time()
    
    ''' Visualización de espacio incrustado '''
    noise = np. random. normal(0,args.noise,tsne_results.shape[0])
    tsne_results[:,0] = tsne_results[:,0] + noise
    noise = np. random. normal(0,args.noise,tsne_results.shape[0])
    tsne_results[:,1] = tsne_results[:,1] + noise
    
    df = pd.DataFrame(tsne_results, columns = ['tsne-one', 'tsne-two'])
    df["target"] = targets
    cols = len(np.unique (targets))
    
    plt.figure(figsize=(10,10))
    #plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="target",
        palette=sns.color_palette("hls", n_colors = cols),
        data=df,
        legend="full",
        alpha=1
    )
    
    plt.legend(['Generated', 'Reals'])
    plt.title("t-SNE for " + args.subset + "\n")
    plt.tight_layout()
    plt.savefig(args.out_image)
    plt.clf(); plt.close("all"); 
    
    logger.info("Figure saved as: " + str(args.out_image) + "\n===================\n")

if __name__ == "__main__": 
    #"""
    sys.argv.append ("--data_path"); sys.argv.append ("Features/he_7k_cd5/"); 
    sys.argv.append ("--num_samples"); sys.argv.append ("-1"); 
    sys.argv.append ("--out_image"); sys.argv.append ("Features/he_7k_cd5/tsne/both.png")
    #sys.argv.append ("--out_results"); sys.argv.append ("/scr/rfonnegr/sources/GANformer/results/cell_s5-000/network-snapshot-002000.pkl"); 
    #sys.argv.append ("--mode"); sys.argv.append ("/scr/rfonnegr/sources/GANformer/results/cell_s5-000/002000_xeeee/"); 
    sys.argv.append ("--subset"); sys.argv.append ("both"); 
    sys.argv.append ("--noise"); sys.argv.append ("0");
    sys.argv.append ("--n_components"); sys.argv.append ("2"); 
    sys.argv.append ("--n_iter"); sys.argv.append ("250"); 
    #sys.argv.append ("--verbose"); sys.argv.append ("1"); 
    #"""
    main()