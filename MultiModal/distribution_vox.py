import sys
import tqdm
from ecapa_tdnn_features import *
sys.path.insert(0, r'../')
sys.path.insert(0, r'../common')
sys.path.insert(0, r'../PlugNetwork')

from common import circulent_binary_embedding
model = circulent_binary_embedding.cbe_random(192)

def compute_distance(P1, P2):
    E1 = extract_speech_embeddings(P1)
    E2 = extract_speech_embeddings(P2)
    BE1 = circulent_binary_embedding.cbe_prediction(model, E1)
    BE2 = circulent_binary_embedding.cbe_prediction(model, E2)
    return np.sum(np.logical_xor(BE1, BE2))/ len(E1)


def compute_embedding(P1):
    E1 = extract_speech_embeddings(P1)
    BE1 = circulent_binary_embedding.cbe_prediction_with_opd(model, E1)
    return BE1

if __name__ == '__main__':
    

    with open('voxceleb_pairs.txt' , 'r+') as f:
         lines = f.readlines()

    mated, non_mated = [] , []
    for l in tqdm.tqdm(lines[-3000 :] + lines[ :3000]):
        a, b, c  = l.split(' ')
        try:
            d = compute_distance(a.strip(), b.strip())
            if c.strip() == '1':
                mated.append(d)
            else:
                non_mated.append(d)
        except Exception as e:
            print('error observed ..', a, b, c)
            continue
        
    print(np.mean(mated), np.mean(non_mated))
    import json
    with open('../dumps/voxceleb_dist_1', 'w+') as f:
        json.dump({'mated' : mated, 'non-mated' : non_mated}, f)

