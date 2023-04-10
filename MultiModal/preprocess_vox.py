import os 
import numpy as np
import itertools
if __name__ == '__main__':
    P ='../dumps/voxceleb/data/'
    D = {}
    for e in os.listdir(P):
        D[e] = []
        for (dirpath, dirnames, filenames) in os.walk(P + e):
            D[e].extend([dirpath + '/' + f for f in filenames])
    
    M = []
    NM = []
    import tqdm
    with open('voxceleb_pairs.txt' , 'w+') as file:
        for e in tqdm.tqdm(D.keys()):
            if len(D[e]) == 0: continue
            D[e] = list(np.random.choice(D[e], max(5, len(D[e])), replace=False))
            for f1, f2 in itertools.combinations(D[e], 2):
                M.append(f'{f1} {f2} 1\n')
        
        M = list(np.random.choice(M, 5000, False))
        F = list(D.keys())
        for i in tqdm.tqdm(range(len(F)-1)):
            for j in range(i+1, len(F)):
                for e, f in itertools.product(D[F[i]], D[F[j]]):
                    NM.append(f'{e} {f} 0\n')
                    
                NM = list(np.random.choice(NM, 5000, False))
        print('done...!')
        file.writelines(M + NM)

    print('done!')

    import json
    with open('voxceleb_ids.txt', 'w+') as f:
        json.dump(D, f)