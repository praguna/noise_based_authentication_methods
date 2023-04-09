import os
import numpy as np
import itertools
P = '../dumps/cfp-dataset/Data/Images/'

if __name__ == '__main__':
    I = os.listdir(P)
    file_names = os.listdir(P +'/'+I[0] + '/' + 'frontal')

    M = []
    NM = []
    with open('cfp_pairs.txt' , 'w+') as file:
        for fname in I:
            for e,f in itertools.combinations(file_names, 2):
                M.append(f'{fname} {fname} {e} {f}\n')

        for e, f in itertools.combinations(I, 2):
            for g,h in itertools.product(file_names, file_names):
                NM.append(f'{e} {f} {g} {h}\n')

        # f.writelines(M + NM)
        file.writelines(list(np.random.choice(M, 5000)) + list(np.random.choice(NM, 5000)))