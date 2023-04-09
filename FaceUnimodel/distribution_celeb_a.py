import sys
import tqdm
sys.path.insert(0, r'../')
sys.path.insert(0, r'../common')

from detect_noise import *
from common import circulent_binary_embedding
path_of_pairs = ".txt"
model = circulent_binary_embedding.cbe_random(2048 * 3)

def compute_distance(P1, P2):
    E1 = detect_mediapipe(P1)
    BE1 = circulent_binary_embedding.cbe_prediction(model, E1)
    E2 = detect_mediapipe(P2)
    BE2 = circulent_binary_embedding.cbe_prediction(model, E2)
    return np.sum(np.logical_xor(BE1, BE2))/ len(E1)

if __name__ == '__main__':
    
    R = '../dumps/img_align_celeba/'
    with open('celeb_pairs.txt' , 'r+') as f:
         lines = f.readlines()

    mated, non_mated = [] , []
    for l in tqdm.tqdm(lines[:10]):
        a, b, c  = l.split(' ')
        try:
            d = compute_distance(R+a.strip(), R+b.strip())
            if c.strip() == '1':
                mated.append(d)
            else:
                non_mated.append(d)
        except:
            print('error observed ..', a, b, c)
            continue
        
    import json
    with open('../dumps/celeb_a_dist', 'w+') as f:
        json.dump({'mated' : mated, 'non-mated' : non_mated}, f)