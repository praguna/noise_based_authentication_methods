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


def compute_plug_noise(P1):
    E1, M1 = inference.compute_embedding_with_distance(P1)
    BE1 = circulent_binary_embedding.cbe_prediction_with_opd(model, E1)
    return BE1

def compute_noise(P1):
    E1 = detect_mediapipe(P1)
    BE1 = circulent_binary_embedding.cbe_prediction_with_opd(media_model, E1)
    return BE1

if __name__ == '__main__':
    
    R = '../dumps/lfw_database/lfw-deepfunneled/lfw-deepfunneled/'
    with open('../dumps/lfw_database/pairs.csv' , 'r+') as f:
         lines = f.readlines()

    mated, non_mated = [] , []
    for l in tqdm.tqdm(lines):
        a, b, c, d  = l.split(',')
        a, b , c, d  = a.strip(), b.strip(), c.strip(), d.strip()
        try:
            f1, f2 = a , a if len(d) == 0 else c
            b = (4 - len(b)) * '0' + b
            n2 = (4 - len(d)) * '0' + d
            if len(d) == 0: n2 = (4 - len(c)) * '0' + c
            # continue
            dis = compute_distance_plug(R+f1+'/'+f1+'_'+b+'.jpg', R+f2+'/'+f2+'_'+n2+'.jpg')
            if len(d) == 0:
                mated.append(dis)
            else:
                non_mated.append(dis)
        except FileNotFoundError as e:
            print(e)
            continue
        except Exception as e:
            # raise e
            print('error observed ..', a, b, c, d, e)
            continue
    
    import json
    with open('../dumps/lfw_dist_p2', 'w+') as f:
        json.dump({'mated' : mated, 'non-mated' : non_mated}, f)
    
    import json
    with open('../dumps/lfw_mean_p2', 'w+') as f:
        json.dump({'coorelation' : X}, f)