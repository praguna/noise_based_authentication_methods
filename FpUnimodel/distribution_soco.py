import sys
import tqdm
sys.path.insert(0, r'../')
sys.path.insert(0, r'../common')

from mdl_features import *
from common import circulent_binary_embedding
path_of_pairs = ".txt"
model = circulent_binary_embedding.cbe_random(192)

extract_embeddings = init_fp_session()

R = '../dumps/SOCOFing/'

def compute_distance(P1, P2, noise = True):
    I = preprocess_test([P1, P2], noise)
    F_norm = extract_embeddings(I)
    E1 = F_norm[0 , :]
    BE1 = circulent_binary_embedding.cbe_prediction_with_opd(model, E1)
    E2 = F_norm[1, : ]
    BE2 = circulent_binary_embedding.cbe_prediction_with_opd(model, E2)
    return np.sum(np.logical_xor(BE1, BE2))/ len(E1)

if __name__ == '__main__':
    

    with open('soco_pairs.txt' , 'r+') as f:
         lines = f.readlines()

    mated, non_mated = [] , []
    for l in tqdm.tqdm(lines[:3000] + lines[-3000:]):
        a, b, c  = l.split(' ')
        try:
            d = compute_distance(R+a.strip(), R+b.strip())
            if c.strip() == '1':
                mated.append(d)
            else:
                non_mated.append(d)
        except Exception as e:
            print('error observed ..', a, b, c)
            continue
        
    import json
    with open('../dumps/soco_dist_4', 'w+') as f:
        json.dump({'mated' : mated, 'non-mated' : non_mated}, f)