#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/12 18:37
# @author : Xujun Zhang

import argparse
import gc
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import reduce
from itertools import chain
from time import time

warnings.filterwarnings('ignore')

def merge_2_df(df1, df2):
    new_df = df1.append(df2, sort=False)
    return new_df

def generate_tmp_csv(src_txt, tmp_csv):
    with open(src_txt, 'r') as f:
            contents = f.read().split('<EOS>\n')[:-1]  # return list
    # for each seed smile
    names, smiles, mw, logp, rb, hba, hbr, halx, similarities, labels, trains = [], [], [], [], [], [], [], [], [], [], []
    for seed, content in enumerate(tqdm(contents)):
        content = content.splitlines()
        # collate
        # each generated smile
        for smi_num, line in enumerate(content):
            line = line.split()
            
            if len(line) == 11:
                n = line[-1]
                s = float(line[7]) #similarity
                if s > 0.4  and not n.endswith("_0"):
                    continue # do not take into accout decoys which are highly similary to a seed molecule
                names.append(n)
                smiles.append(line[0])
                for i, lis in enumerate([mw, logp, rb, hba, hbr, halx, similarities, labels, trains]):
                    value = float(line[i + 1]) #because 0 is smile column
                    lis.append(value)
            else:
                print(f'error for {line}')

    df = pd.DataFrame([names, smiles, mw, logp, rb, hba, hbr, halx, similarities, labels, trains]).T
    df.columns = ['name', 'smile', 'mw', 'logp', 'rb', 'hba', 'hbr', 'halx', 'similarity', 'label', 'train']
    print(f"in dataset: {df.shape[0]}")
    df.loc[:, "mw"] = df.loc[:, "mw"].astype(np.float16) #mw
    df.loc[:, "logp"] = df.loc[:, "logp"].astype(np.float16) #logp
    df.loc[:, "rb"] = df.loc[:, "rb"].astype(np.uint8) #rotatable bonds
    df.loc[:, "hba"] = df.loc[:, "hba"].astype(np.uint8) #hba
    df.loc[:, "hbr"] = df.loc[:, "hbr"].astype(np.uint8) #hbr
    df.loc[:, "halx"] = df.loc[:, "halx"].astype(np.uint8) #halx
    df.loc[:, "similarity"] = df.loc[:, "similarity"].astype(np.float16) #similarity
    df.loc[:, "label"] = df.loc[:, "label"].astype(np.uint8) #labels
    df.loc[:, "train"] = df.loc[:,"train"].astype(np.uint8) #trains
    df.to_csv(tmp_csv, index=False)

def filtering(mol_name):
    
    global df
    property_ranges = {'mw':40, 'logp':1.5, 'rb':1, 'hba':1, 'hbr':1, 'halx':1}
    df_tmp = df[df['name'].str.startswith(f"{mol_name}_")]
    conditions = []
    seed_idx = df_tmp[df_tmp['name'].str.endswith("_0")].index.values[0]
    for col in ['mw', 'logp', 'rb', 'hba', 'hbr', 'halx']:
        val = df_tmp.loc[seed_idx, col]
        conditions_l = df_tmp[col] >= val - property_ranges[col]
        conditions_r = df_tmp[col] <= val + property_ranges[col]
        conditions.append(conditions_l)
        conditions.append(conditions_r)

    condition = reduce(lambda x, y: x & y, conditions, True)
    return df_tmp[condition].index.values.tolist()

if __name__ == '__main__':
    # init
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str, default='/root/zxj/du/total_workflow')
    argparser.add_argument('--target', type=str, default='du')
    argparser.add_argument('--src_path', type=str, default='2_propterties_filter')
    argparser.add_argument('--dst_path', type=str, default='2_propterties_filter')
    argparser.add_argument('--src_file', type=str, default='dst_smi.txt')
    argparser.add_argument('--tmp_file', type=str, default='property_unfiltered.csv')
    argparser.add_argument('--dst_file', type=str, default='property_filtered.csv')
    args = argparser.parse_args()
    #
    src_txt = f'{args.src_file}'
    tmp_csv = f'{args.tmp_file}'
    dst_csv = f'{args.dst_file}'
    if not os.path.exists(tmp_csv):
        print('collate data from txt to csv....')
        generate_tmp_csv(src_txt, tmp_csv)
    # read csv
    print('read data from csv file')
    df = pd.read_csv(tmp_csv)
    print(df.head())
    names = df.loc[:, 'name'].values
    names = set([i.split("_")[0] for i in names])
    start = time()
    cpus = 24
    with Pool(cpus) as p:
        indices = p.map(filtering, names)
    finish = time()
    indices = list(chain(*indices)) #flatten list of lists
    print(f"before filtering: {df.shape[0]}")
    df = df.loc[indices, :]
    print(f"after filtering: {df.shape[0]}")

    df.to_csv(dst_csv, index=False)
    print('end filtering')
    print(f'filtering takes: {finish - start}')