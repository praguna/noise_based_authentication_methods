import sys
import tqdm
import os
import numpy as np
sys.path.insert(0, r'../')
sys.path.insert(0, r'../common')
sys.path.insert(0, r'../FpUnimodel')
sys.path.insert(0, r'../MultiModal')

from FpUnimodel import distribution_soco
from FpUnimodel import imageprocessing
from FpUnimodel import mdl_features
from MultiModal import distribution_vox

import json
# '../MultiModal/voxceleb_pair.txt'
with open('../MultiModal/voxceleb_ids.txt') as f:
    noise_dict = json.load(f)

noise_keys = list(sorted(noise_dict.keys()))


def corresponding_voice(fname, other_path = None):
    id = int(fname.split('_')[0].strip())
    vid = noise_keys[id]
    c = np.random.choice(noise_dict[vid] , 2, False)
    if other_path and other_path == c[0]: return c[1]
    return c[0]

def fvc_example_generator():
    R = '../dumps/fp_datasets/FVC2004/Dbs/DB3_A/'
    with open('../dumps/fp_datasets/FVC2004/Dbs/index_a.MFA' , 'r+') as f:
         lines = [l + ' 0' for l in f.readlines()]
        
    with open('../dumps/fp_datasets/FVC2004/Dbs/index_a.MFR' , 'r+') as f:
        lines.extend([l + ' 1' for l in f.readlines()])

    for l in tqdm.tqdm(lines[ : 500] + lines[-500 :]):
        a1, b1, c1  = l.split(' ')
        a = corresponding_voice(a1.strip())
        b = corresponding_voice(b1.strip(), a)
        yield str(R+a1.strip()), str(R+b1.strip()), c1, str(a), str(b)

def extract_noise(P):
    return  distribution_vox.compute_embedding(P)

def extract_feature(P):
    I = imageprocessing.preprocess_test([P], False)
    F_norm = distribution_soco.extract_embeddings(I)
    return F_norm[0:]


if __name__ == "__main__":
    R = '../dumps/SOCOFing/'
    with open('../FpUnimodel/soco_pairs_ids.txt' , 'r+') as f:
         lines = f.readlines()

    mated, non_mated = [] , []
    for a1, b1, c1, a, b in fvc_example_generator():
        print(a1, b1, c1, a , b)
        exit(0)
        try:
            d = distribution_soco.compute_distance(a1.strip(), b1.strip(), False)
            if c1.strip() == '1':
                mated.append(d)
            else:
                non_mated.append(d)
            print(d)
        except Exception as e:
            print('error observed ..', a, b, c1)
            continue
    pass