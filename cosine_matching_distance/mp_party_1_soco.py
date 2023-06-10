import sys
import tqdm
import os
import numpy as np
sys.path.insert(0, r'../')
sys.path.insert(0, r'../common')
sys.path.insert(0, r'../FpUnimodel')

from FpUnimodel import distribution_soco
from FpUnimodel import imageprocessing
from FpUnimodel import mdl_features
from common import circulent_binary_embedding


def corresponding_index(path, other_path = None):
    fname = path.split('/')[-1]
    id = fname.split('__')[0].strip()
    h = 'Left' if path.find('Left')!=-1 else 'Right'
    p = [path[:-len(fname)] + e for e in os.listdir(path[:-len(fname)]) if e.startswith(id + '__') and e.find('little_finger')!=-1 and e.find(h)!=-1] 
    c = np.random.choice(p , 2)
    if other_path and other_path == c[0]: return c[1]
    return c[0]

def soco_example_generator():
    R = '../dumps/SOCOFing/'
    with open('../FpUnimodel/soco_pairs_ids.txt' , 'r+') as f:
         lines = f.readlines()

    for l in tqdm.tqdm( lines[ :500] + lines[-500 :]):
        a1, b1, c1  = l.split(' ')
        a = corresponding_index(R+a1.strip())
        b = corresponding_index(R+b1.strip(), a)
        yield str(R+a1.strip()), str(R+b1.strip()), c1, str(a), str(b)

def extract_noise(P):
    I = imageprocessing.preprocess_test([P], False)
    F_norm = distribution_soco.extract_embeddings(I)
    E1 = F_norm[0 , :]
    BE1 = circulent_binary_embedding.cbe_prediction_with_opd(distribution_soco.model, E1)
    return BE1

def extract_feature(P):
    I = imageprocessing.preprocess_test([P], False)
    F_norm = distribution_soco.extract_embeddings(I)
    return F_norm[0:]


if __name__ == "__main__":
    R = '../dumps/SOCOFing/'
    with open('../FpUnimodel/soco_pairs_ids.txt' , 'r+') as f:
         lines = f.readlines()

    mated, non_mated = [] , []
    for a1, b1, c1, a, b in soco_example_generator():
        try:
            d = distribution_soco.compute_distance(R+a1.strip(), R+b1.strip(), False)
            if c1.strip() == '1':
                mated.append(d)
            else:
                non_mated.append(d)
            print(d)
        except Exception as e:
            print('error observed ..', a, b, c1)
            continue
    pass