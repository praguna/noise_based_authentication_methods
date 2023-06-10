import sys
import tqdm
import os
import numpy as np
sys.path.insert(0, r'../')
sys.path.insert(0, r'../common')
sys.path.insert(0, r'../FaceUnimodel')
sys.path.insert(0, r'../PlugNetwork')

from PlugNetwork import inference
from FaceUnimodel import distribution_lfw

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

def lfw_face_example_generator():
    R = '../dumps/lfw_database/lfw-deepfunneled/lfw-deepfunneled/'
    with open('../dumps/lfw_database/pairs.csv' , 'r+') as f:
         lines = f.readlines()

    for l in tqdm.tqdm(lines[ :500] + lines[-500 :]):
        a, b, c, d  = l.split(',')
        a, b , c, d  = a.strip(), b.strip(), c.strip(), d.strip()
        f1, f2 = a , a if len(d) == 0 else c
        b = (4 - len(b)) * '0' + b
        n2 = (4 - len(d)) * '0' + d
        if len(d) == 0: n2 = (4 - len(c)) * '0' + c
            # continue
        x1, x2 = R+f1+'/'+f1+'_'+b+'.jpg' ,  R+f2+'/'+f2+'_'+n2+'.jpg'
        if 'name_imagenum1.jpg' in x1: continue
        yield str(x1.strip()), str(x2.strip()), str(int(len(d) == 0)), str(x1.strip()), str(x2.strip())


def celeb_a_face_example_generator():
    R = '../dumps/img_align_celeba/'
    with open('../FaceUnimodel/celeb_pairs.txt' , 'r+') as f:
         lines = f.readlines()

    for l in tqdm.tqdm(lines[ :500] + lines[-500 :]):
        a, b, c = l.split(' ')
        x1, x2 = R+a.strip(), R+b.strip()
        yield str(x1.strip()), str(x2.strip()), c.strip(), str(x1.strip()), str(x2.strip())


def celeb_a_cfp_example_generator():
    R = '../dumps/cfp-dataset/Data/Images/'
    with open('../FaceUnimodel/cfp_pairs.txt' , 'r+') as f:
         lines = f.readlines()

    for l in tqdm.tqdm(lines[ :100] + lines[-100 :]):
        a, b, c, d = l.split(' ')
        x1, x2 =  R +'/'+a.strip() + '/' + 'frontal/' + c.strip(), R +'/'+b.strip() + '/' + 'frontal/' + d.strip()
       
        yield str(x1.strip()), str(x2.strip()), str(int(a.strip() == b.strip())), str(x1.strip()), str(x2.strip())

def extract_noise(P):
    return  distribution_lfw.compute_plug_noise(P).flatten()

def extract_feature(P):
    return inference.compute_id(P)


if __name__ == "__main__":
    # R = '../dumps/lfw_database/lfw-deepfunneled/lfw-deepfunneled/'
    # with open('../dumps/lfw_database/pairs.csv' , 'r+') as f:
    #      lines = f.readlines()

    mated, non_mated = [] , []
    for a1, b1, c1, a, b in celeb_a_cfp_example_generator():
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