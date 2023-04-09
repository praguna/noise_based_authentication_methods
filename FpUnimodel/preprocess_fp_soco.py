import os
import itertools
import numpy as np


P = '../dumps/SOCOFing/'
if __name__  == "__main__":
    L = ['Real', 'Altered']
    sd1 = os.listdir(P + '/Altered')
    files = os.listdir(P + '/Real')
    files = ['Real/' + x for x in files if x.find('little_finger')!=-1]
    files = [e for e in  files if int(e.split('__')[0].split('/')[-1]) <= 100]
    for p in sd1: 
        try:
            e = os.listdir(P +  'Altered/' + p)
            e = ['Altered/' + p + '/'  + x for x in e if x.find('little_finger')!=-1]
            e = [x for x in  e if int(x.split('__')[0].split('/')[-1]) <= 100]
            files.extend(e)
        except Exception as e:
            print('error : ignoring', e)
            pass

    print(len(files))
    M = []
    NM = []
    with open('soco_pairs.txt' , 'w+') as file:
        for e, f in itertools.combinations(files, 2):
            id1 = e.split('__')[0].split('/')[-1]
            id2 = f.split('__')[0].split('/')[-1]
            # print(e, f, id1, id2)
            same_hand = ((e.find('Left')!=-1 and f.find('Left')!=-1) or (e.find('Right')!=-1 and f.find('Right')!=-1))
            # same_finger = ((e.find('little_finger')!=-1 and f.find('little_finger')!=-1) or (e.find('index_finger')!=-1 and f.find('index_finger')!=-1))
            same_finger = ((e.find('little_finger')!=-1 and f.find('little_finger')!=-1) or (e.find('index_finger')!=-1 and f.find('index_finger')!=-1))
            if id1 == id2 and same_hand and same_finger:
                M.append(f'{e} {f} {1}\n')
            else:
                NM.append(f'{e} {f} {0}\n')

        # file.writelines(M + NM)
        file.writelines(list(np.random.choice(M, 5000)) + list(np.random.choice(NM, 5000)))