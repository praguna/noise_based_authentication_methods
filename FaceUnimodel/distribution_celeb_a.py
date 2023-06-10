import sys
import tqdm
sys.path.insert(0, r'../')
sys.path.insert(0, r'../common')
sys.path.insert(0, r'../PlugNetwork')

from detect_noise import *
from common import circulent_binary_embedding
from PlugNetwork import inference
path_of_pairs = ".txt"
media_model = circulent_binary_embedding.cbe_random(2048 * 3)
model = circulent_binary_embedding.cbe_random(512)
X = []

def compute_distance(P1, P2):
    E1 = detect_mediapipe(P1)
    BE1 = circulent_binary_embedding.cbe_prediction(media_model, E1)
    E2 = detect_mediapipe(P2)
    BE2 = circulent_binary_embedding.cbe_prediction(media_model, E2)
    return np.sum(np.logical_xor(BE1, BE2))/ len(E1)

def compute_distance_plug(P1, P2):
    E1, M1 = inference.compute_embedding_with_distance(P1)
    BE1 = circulent_binary_embedding.cbe_prediction(model, E1)
    E2, M2 = inference.compute_embedding_with_distance(P2)
    BE2 = circulent_binary_embedding.cbe_prediction(model, E2)
    X.extend([float(M1), float(M2)])
    return np.sum(np.logical_xor(BE1, BE2))/ len(E1)


def compute_noise(P1):
    E1, M1 = inference.compute_embedding_with_distance(P1)
    BE1 = circulent_binary_embedding.cbe_prediction(media_model, E1)
    return BE1

def compute_plug_noise(P1):
    E1 = detect_mediapipe(P1)
    BE1 = circulent_binary_embedding.cbe_prediction(model, E1)
    return BE1

if __name__ == '__main__':
    
    R = '../dumps/img_align_celeba/'
    with open('celeb_pairs.txt' , 'r+') as f:
         lines = f.readlines()

    mated, non_mated = [] , []
    for l in tqdm.tqdm(lines[:3000] + lines[-3000:]):
        a, b, c  = l.split(' ')
        try:
            d = compute_distance_plug(R+a.strip(), R+b.strip())
            if c.strip() == '1':
                mated.append(d)
                # print(d)
            else:
                non_mated.append(d)
                # print(d)
        except Exception as e:
            # raise e
            print('error observed ..', a, b, c, e)
            continue

    print(np.average(mated), np.average(non_mated))
        
    import json
    with open('../dumps/celeb_a_dist_p2', 'w+') as f:
        json.dump({'mated' : mated, 'non-mated' : non_mated}, f)
    

    import json
    with open('../dumps/celeb_a_mean_p2', 'w+') as f:
        json.dump({'coorelation' : X}, f)