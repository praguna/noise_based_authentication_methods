import itertools
import numpy as np
if __name__ == '__main__':
    p = '../dumps/celeb_a_annotations/identity_CelebA.txt'
    with open(p, 'r')  as f:
        ids = f.readlines()
    
    D = {}
    D1 = {}
    for e in ids:
        a, b = e.split(' ')
        D1[a.strip()] = b.strip()
        if b.strip() in D:
            D[b.strip()].append(a.strip())
        else:
            D[b.strip()] = [a.strip()]

    p = '../dumps/celeb_a_annotations/list_eval_partition.txt'
    with open(p, 'r') as f:
        partition = f.readlines()
        partition = [x for x in partition if x.split(' ')[1].strip() == '1']
    
    # select 10 samples per class for first 500 unique identities
    p = {}
    for e in partition:
        a, b = e.split()
        if len(p) == 100:
            break
        if D1[a.strip()] in p:
            continue
        else:
            p[D1[a.strip()]] = 1
    
    files = []
    ids = []
    for k, v in p.items():
        files.append(D[k][:8])
        ids.append(k)
    
    M = []
    NM = []
    with open('celeb_pairs.txt' , 'w+') as f:
        for i in range(0, len(files)-1):
            for pair in itertools.combinations(files[i], 2):
                M.append(f'{pair[0]} {pair[1]} 1\n')

        for i in range(0, len(files)-1):
            for j in range(i, len(files)):
                for pair in itertools.product(files[i], files[j]):
                    NM.append(f'{pair[0]} {pair[1]} 0\n')

        # f.writelines(M, NM)
        f.writelines(list(np.random.choice(M, 5000)) + list(np.random.choice(NM, 5000)))
        

    # import json
    # data = {'F' : files, 'id' : ids}
    # with open('test_files.json', 'w+') as f:
    #     json.dump(data, f)

    print('done!!')
