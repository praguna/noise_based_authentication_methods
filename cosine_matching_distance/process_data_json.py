if __name__ == "__main__":
    import json, matplotlib.pyplot as plt, numpy as np
    with open('celeb_scores.json', 'r+') as f:
        obj = json.load(f)

    print(len(obj['mated']['protocol']))
    def get_metrics(t):
        '''returns FAR, FRR'''
        FRR, FRR1, FAR ,FAR1= 0, 0, 0, 0
        M = obj['mated']
        NM = obj['non-mated']
        arr1, arr2 = M['local'], M['protocol']
        for e1, e2 in zip(arr1, arr2):
            if e1 < t: FRR += 1
            if e2 < t: FRR1 += 1
        
        arr1, arr2 = NM['local'], NM['protocol']
        for e1, e2 in zip(arr1, arr2):
            if e1  >= t: FAR+=1
            if e2  >= t: FAR1+=1

        # print(FRR, FAR, t)
        return FRR / len(M['local']),  FRR1 /  len(M['local']), FAR /  len(NM['local']), FAR1 /  len(NM['local'])


    b = True
    FRR_l , FAR_l = [] , []
    FRR_p , FAR_p = [] , []
    FRR_0, FRR_01 = [] , []
    err0 = []
    err1 = []
    i = 0
    for t in np.arange(0.0, 1.0, 0.001):
        FRR, FRR1, FAR, FAR1 = get_metrics(t)
        FRR_l.append(FRR)
        FAR_l.append(FAR)
        FRR_p.append(FRR1)
        FAR_p.append(FAR1)
        err0.append(abs(FAR - FRR))
        err1.append(abs(FAR1 - FRR1))
        # print(FAR)
        if abs(FAR - 0.002) < 1e-4: FRR_0.append(FRR)
        if abs(FAR1 - 0.002) < 1e-4 : FRR_01.append(FRR1)
        i+=1

    print(min(FRR_0), min(FRR_01))
    print(FRR_l[np.argmin(err0)] , FRR_p[np.argmin(err1)])
    # plot corrected values
    # plt.plot(np.array(FAR_l) * 100, np.array(FRR_l) * 100  ,  linestyle='dashed' , color = 'r', label='without protocol')
    # plt.plot(np.array(FAR_p) * 100, np.array(FRR_p) * 100 ,  linestyle='dashed', color = 'b', label='with secure protocol')
    # plt.xlabel('FMR in(%)')
    # plt.ylabel('FNMR in(%)')
    # # plt.title('FVCDba1 DET Plot')
    # plt.title('FVC DBA3 DET Plot')
    # plt.xlim(left=0.1)
    # plt.ylim(top=20)
    # plt.legend()
    # plt.savefig('../dumps/fvc_dba3_det.png')