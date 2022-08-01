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
from class_base.property_filter_base import properties_filer

warnings.filterwarnings('ignore')


def merge_2_df(df1, df2):
    new_df = df1.append(df2, sort=False)
    return new_df


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
    # path = f'{args.path}/{args.target}'
    # src_path = f'{path}/{args.src_path}'
    # dst_path = f'{path}/{args.dst_path}'
    src_txt = f'{args.src_file}'
    tmp_csv = f'{args.tmp_file}'
    dst_csv = f'{args.dst_file}'
    if not os.path.exists(tmp_csv):
        print('collate data from txt to csv....')
        pd.DataFrame(['name', 'smile', 'mw', 'logp', 'rb', 'hba', 'hbr', 'halx', 'similarity', 'label', 'train']).T.to_csv(
            tmp_csv, index=False, header=None)
        # get smiles
        with open(src_txt, 'r') as f:
            contents = f.read().split('<EOS>\n')[:-1]  # return list
        # for each seed smile
        for seed, content in enumerate(tqdm(contents)):
            # seed = seed + 1297
            content = content.splitlines()
            # collate
            names, smiles, mw, logp, rb, hba, hbr, halx, similarities, labels, trains = [], [], [], [], [], [], [], [], [], [], []
            # each generated smile
            for smi_num, line in enumerate(content[:]):
                line = line.split()
                if len(line) == 11:
                    for i, lis in enumerate([smiles, mw, logp, rb, hba, hbr, halx]):
                        if i == 0:
                            value = line[i]
                        else:
                            value = float(line[i])
                        lis.append(value)

                    # float similarity
                    similarities.append(float(line[7]))
                    labels.append(float(line[8]))
                    trains.append(float(line[9]))
                    names.append(line[-1])
                else:
                    print(f'error for {line}')
            # trans to df
            df = pd.DataFrame([names, smiles, mw, logp, rb, hba, hbr, halx, similarities, labels, trains]).T
            print(df.head())
            df_seed = pd.DataFrame(df.iloc[0, :]).T  # get seed
            df = df[df.iloc[:, -3] <= 0.4]
            df.sort_values(by=8, inplace=True, ascending=True)
            df = df_seed.append(df, sort=False)
            #reduce memory consumption
            del df_seed
            df.iloc[:, 2] = df.iloc[:, 2].astype(np.float16) #mw
            df.iloc[:, 3] = df.iloc[:, 3].astype(np.float16) #logp
            df.iloc[:, 4] = df.iloc[:, 4].astype(np.float16) #rb
            df.iloc[:, 5] = df.iloc[:, 5].astype(np.uint8) #hba
            df.iloc[:, 6] = df.iloc[:, 6].astype(np.uint8) #hbr
            df.iloc[:, 7] = df.iloc[:, 7].astype(np.uint8) #halx
            df.iloc[:, 8] = df.iloc[:, 8].astype(np.float16) #similarity
            df.iloc[:, 9] = df.iloc[:, 9].astype(np.uint8) #labels
            df.iloc[:, 10] = df.iloc[:, 10].astype(np.uint8) #trains
            df.to_csv(tmp_csv, index=False, header=None, mode='a')
    del contents
    del df
    del names
    del smiles
    del mw
    del logp
    del rb
    del hba
    del hbr
    del halx
    del similarities
    del labels
    del trains
    gc.collect()
    # read csv
    print('read data from csv file')
    df = pd.read_csv(tmp_csv) #, encoding='utf-8')
    print(df.head())
    my_filter = properties_filer(df=df)
    print('start filter....')
    result_dfs = []
    for name in my_filter.names:
        tmp_df = my_filter.name2filter(name)
        result_dfs.append(tmp_df)
    del tmp_df #free memory
    print('drop nan')
    result_dfs = list(filter(lambda x: x is not None, result_dfs))
    # merge df
    print('start merging')
    new_df = reduce(merge_2_df, result_dfs)
    print(new_df.head())
    new_df.columns = df.columns
    # write
    print('output to csv')
    new_df.to_csv(dst_csv, index=False)
    print('end filtering')
