import argparse
from functools import partial
from itertools import chain
import gc
from time import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from scipy import sparse

from multiprocessing import Pool
# from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP

import shutil
from collections import ChainMap
import plotly.express as px

def write_ecfp_full(fps, fname):
    
        if len(fps) < 1:
            raise ValueError("No fingerprints to save")

        print("writing ecfp in file")
        with open(fname, "w") as fout:
            
            header = ["name"] + [str(i) for i in range(FP_DIM)] + ["label", "train"]
            header = ",".join(header) + "\n"
            fout.write(header)

            while len(fps):
                fp = fps.pop()
                if fp is not None:
                    fp = [str(i) for i in fp]
                    fp = ",".join(fp) + "\n"
                    fout.write(fp)

def write_ecfp_sparse(fps, fname):
    if len(fps) < 1:
        raise ValueError("No fingerprints to save")
    
    print("writing ecfp in file")
    with open(fname, "w") as fout:
        fout.write("name,positions,label,train\n")

        for fp in tqdm(fps):
            if fp is not None:

                fout.write(fp[0]) #name
                fout.write(",")

                positions = []
                for i in range(FP_DIM):
                    if int(fp[i+1]) == 1:
                        positions.append(str(i))

                positions_str = "-".join(positions)
                fout.write(positions_str)
                fout.write(",")
                fout.write(str(fp[-2])) #label
                fout.write(",")
                fout.write(str(fp[-1])) #train
                fout.write("\n")

def read_ecfp_sparse(fname):
    
    ecfps = [] #name, ecfp, label, train
    print("reading ecfp from sparse format . . . ")
    with open(fname, "r") as fin:
        header = fin.readline().strip()
        c = 0
        line = fin.readline().strip()
        while line:
            line = line.split(",")
            name = line[0]
            fp_sparse = line[1].split("-")
            label = line[2]
            train = line[3]
            fp = [0] * FP_DIM
            for i in fp_sparse:
                fp[int(i)] = 1
            
            ecfps.append([name] + fp + [label, train])
            line = fin.readline().strip()
            c += 1
            if c == 10000:
                print(f"{c} ecfps loaded")
        
    df = pd.DataFrame(ecfps, columns=['name'] + [i for i in range(FP_DIM)] + ['label', 'train'])
    for i in range(FP_DIM):
        df[i] = df[i].astype(np.uint8)

    return df 

def cal_ecfp_chunk(indices):
    ecfps = []
    for i in indices:
        fp = cal_ecfp(i)
        if fp is not None:
            ecfps.append(fp)
    return ecfps

def cal_ecfp(idx):
    global names
    global smiles
    global labels
    global trains
    lig_name, lig_smile, lig_label, lig_train = names[idx], smiles[idx], labels[idx], trains[idx]
    tmp_mol = Chem.MolFromSmiles(lig_smile)
    try:
        ecfp = AllChem.GetHashedMorganFingerprint(tmp_mol, 2, FP_DIM)
        fp_array = np.zeros((0, ), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(ecfp, fp_array)
        fp_array = fp_array.nonzero()[0].tolist()
        tmp = [lig_name] + fp_array + [lig_label] + [lig_train]
        return tmp
    except:
        return None

def cal_reduction(fps):
    print('cal dimensionality reduction')
    n_jobs = max(1, os.cpu_count() - 1)
    # reduction = TSNE(n_components=2, n_jobs=n_jobs, random_state=42, perplexity=100, verbose=True, learning_rate=1000, n_iter=2000)
    reduction = UMAP(n_components=2, verbose=True, n_neighbors=100, low_memory=True) #
    y = reduction.fit_transform(fps)
    return y

def clustering_parallell(grid_unit):
    global dims
    
    min_1, min_2 = dims.min(axis=0)
    max_1, max_2 = dims.max(axis=0)

    N = dims.shape[0]
    cpus = os.cpu_count()

    chunk_size = N // cpus

    chunk_is = [i for i in range(cpus)]

    with Pool(cpus) as p:
        bin_dicts = p.map(partial(binarize, chunk_size=chunk_size,\
        min_1=min_1, min_2=min_2,\
        max_1=max_1, max_2=max_2,\
        grid_unit=grid_unit), chunk_is)
    
#     print(bin_dicts)
    indices = dict(ChainMap(*bin_dicts))
#     print("indices map: ", indices)
    indices = list(indices.values())
    return indices

def binarize(i, chunk_size, min_1, min_2, max_1, max_2, grid_unit):
    global dims
    grid_1 = np.linspace(min_1, max_1, grid_unit + 1)
    grid_2 = np.linspace(min_2, max_2, grid_unit + 1)
    start, end = i*chunk_size, (i+1)*chunk_size
#     print("inside dims: ", dims)
    bins_1 = np.searchsorted(grid_1, dims[start:end, 0], side='left')
    bins_2 = np.searchsorted(grid_2, dims[start:end, 1], side='left')
#     print("bins_1: ", bins_1)
#     print("bins_2: ", bins_2)
    
    bin2count = {}
    for k, (b1, b2) in enumerate(zip(bins_1, bins_2)):
        bin2count.setdefault((b1, b2), k)
#     print("bin2count: ", bin2count)
    
    return bin2count
    
#some parameters
FP_DIM = 2048 #initially is 2048
# init
argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='/root/zxj/du/total_workflow')
argparser.add_argument('--target', type=str, default='mpro')
argparser.add_argument('--src_path', type=str, default='2_propterties_filter')
argparser.add_argument('--dst_path', type=str, default='2_propterties_filter')
argparser.add_argument('--src_file', type=str, default='property_filtered.csv')
argparser.add_argument('--ecfp_file', type=str, default='ecfp.csv')
argparser.add_argument('--reduced_file', type=str, default='reduced.csv')
argparser.add_argument('--dst_file', type=str, default='filtered.csv')
argparser.add_argument('--decoys_size', type=int, default=100)
argparser.add_argument('--grid_unit', type=int, default=1000)
args = argparser.parse_args()
#
src_csv = f'{args.src_file}'
ecfp_csv = f'{args.ecfp_file}'
reduced_csv = f'{args.reduced_file}'
dst_csv = f'{args.dst_file.split(".")[0]}_{args.grid_unit}.txt'


df = pd.read_csv(src_csv)
# df = df.iloc[:10000]
smiles = df.smile.values
names = df.name.values
trains = df.train.values.astype(np.uint8)
labels = df.label.values.astype(np.uint8)
decoys_size = args.decoys_size
grid_unit = args.grid_unit

del df

# ecfp and dimensionality reduction
if True:
    start = time()
    print('dimensionality reduction')
    cpus = os.cpu_count() - 1
    chunks = np.array_split(range(len(names)), cpus)
    print(f"{cpus} cpu(s) will be used")
    with Pool(cpus) as p:
        fps = p.map(cal_ecfp_chunk, chunks)
    fps = list(chain(*fps)) #unfold first level of nesting
    finish = time()
    print(f"ecfp computing takes {finish - start}")

    # fps = np.array(fps)
    names = []
    labels = []
    train = []
    ecfps = np.zeros((len(fps), FP_DIM), dtype=np.int8)
    print("filling ecfp matrix")
    for i, row in tqdm(enumerate(fps)):
        names.append(row[0])
        labels.append(row[-2])
        train.append(row[-1])
        ones_positions = row[1:-2] #fingerprints generates in sparse format
        for p in ones_positions:
            ecfps[i, p] = 1

    del fps
    print(f"fingerprints shape: {len(ecfps[0])}")

    start = time()
    ecfps = sparse.lil_matrix(ecfps)
    y = cal_reduction(ecfps)
    finish = time()
    print(f'dimensionality reduction takes {finish - start}')
    
    df = pd.DataFrame(y, columns=['dim_1', 'dim_2'])
    df['name'] = names
    df['label'] = labels
    df['train'] = train
    df.to_csv(reduced_csv, index=False)

    del df
    del y
    del names
    del ecfps
    del labels
    del trains
    gc.collect()

#grid filtering
if True:
    start = time()
    print('grid filtering')
    df = pd.read_csv(reduced_csv)
    df_ac = df[df['train'] == 1]
    df_decoys = df[df['train'] == 0]
    target_num = decoys_size * len(df_ac)
    if len(df_decoys) < target_num:
        print('no need grid filtering')
        shutil.copy(src=src_csv, dst=dst_csv)
    else:
        dims = df_decoys.loc[:, ['dim_1', 'dim_2']].values
        indices_filtered = clustering_parallell(grid_unit)
        df_decoys = df_decoys.iloc[indices_filtered, :]
        df_decoys.to_csv(dst_csv, index=False)
        print("decoys in the final file: ", df_decoys.shape[0])
    finish = time()
    print(f'grid filtering takes {finish - start}')
    gc.collect()