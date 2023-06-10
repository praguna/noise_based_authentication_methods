if __name__ == "__main__":
    import json, matplotlib.pyplot as plt, numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score, det_curve

    with open('celeb_scores.json', 'r+') as f:
        obj = json.load(f)

    labels = [1 for _ in range(len(obj['mated']['local']))] + [0 for _ in range(len(obj['non-mated']['local']))]
    pred_r = [abs(e) for e in obj['mated']['local']] + [abs(e) for e in obj['non-mated']['local']]
    pred_p = [abs(e) for e in obj['mated']['protocol']] + [abs(e) for e in obj['non-mated']['protocol']]
    pred_p = [abs(e) for e in obj['mated']['protocol']] + [abs(e) for e in obj['non-mated']['protocol']]

    # def plot_curves():
    fpr, tpr, t = roc_curve(labels, pred_r)
    for e, f in zip(fpr, tpr):
        if abs(e - 0.01) < 1e-4: print(e, f)
    # print(fpr, tpr)
    auc_0 = roc_auc_score(labels, pred_r)
    auc_1 = roc_auc_score(labels, pred_p)
    print(auc_0, auc_1)
    plt.plot(fpr, tpr,  linestyle='dashed' , color = 'r', label='without protocol')
    fpr, tpr, t = roc_curve(labels, pred_p)
    for e, f in zip(fpr, tpr):
        if abs(e - 0.01) < 1e-4: print(e, f)
    plt.plot(fpr, tpr,  linestyle='dashed' , color = 'b', label='with secure protocol')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CelebA ROC Plot')
    plt.legend()
    plt.savefig('../dumps/celeb_roc.png')

    # far, frr, t = det_curve(labels, pred_r)
    # plt.plot(far, frr, linestyle='dashed' , color = 'r', label='without protocol')
    # far, frr, t  = det_curve(labels, pred_p)
    # plt.plot(far, frr,  linestyle='dashed' , color = 'b', label='with secure protocol')
    # plt.xlabel('False Match Rate')
    # plt.ylabel('False Non-Match Rate')
    # plt.title('FVCDBA2 DET Plot')
    # plt.xlim(left=0.1 * 0.01)
    # plt.savefig('../dumps/fvcdba2_det_func.png')

    # def get_metrics(t):
    #     '''returns FAR, FRR'''
    #     FRR, FRR1, FAR ,FAR1= 0, 0, 0, 0
    #     M = obj['mated']
    #     NM = obj['non-mated']
    #     arr1, arr2 = M['local'], M['protocol']
    #     for e1, e2 in zip(arr1, arr2):
    #         if e1 < t: FRR += 1
    #         if e2 < t: FRR1 += 1
        
    #     arr1, arr2 = NM['local'], NM['protocol']
    #     for e1, e2 in zip(arr1, arr2):
    #         if e1  >= t: FAR+=1
    #         if e2  >= t: FAR1+=1

    #     # print(FRR, FAR, t)
    #     return FRR / len(M['local']),  FRR1 /  len(M['local']), FAR /  len(NM['local']), FAR1 /  len(NM['local'])


    # b = True
    # FRR_l , FAR_l = [] , []
    # FRR_p , FAR_p = [] , []
    # FRR_0, FRR_01 = [] , []
    # i = 0
    # for t in np.arange(0.0, 1.0, 0.001):
    #     FRR, FRR1, FAR, FAR1 = get_metrics(t)
    #     FRR_l.append(FRR)
    #     FAR_l.append(FAR)
    #     FRR_p.append(FRR1)
    #     FAR_p.append(FAR1)
    #     if abs(FAR - FRR) < 1e-4: print('A', FAR )
    #     if FAR1 == FRR1: print('B', FAR1 )
    #     if FAR == 0: FRR_0.append((FRR))
    #     if FAR1 == 0: FRR_01.append((FRR))
    #     i+=1

    # print(np.min(FRR_01) , np.min(FRR_0))
    # # plot corrected values
    # plt.plot(np.array(FAR_l) * 100, np.array(FRR_l) * 100  ,  linestyle='dashed' , color = 'r', label='without protocol')
    # plt.plot(np.array(FAR_p) * 100, np.array(FRR_p) * 100 ,  linestyle='dashed', color = 'b', label='with secure protocol')
    # plt.xlabel('FAR in(%)')
    # plt.ylabel('FRR in(%)')
    # plt.title('SocoFing DET Plot')
    # plt.xlim(left=0.1)
    # # plt.ylim(top=20)
    # plt.legend()
    # plt.savefig('../dumps/soco1_det.png')