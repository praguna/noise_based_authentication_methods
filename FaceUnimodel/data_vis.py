from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    # P = '../dumps/voxceleb_dist_1' # datset json path
    # import json
    # with open(P, 'r') as f:
    #     obj = json.load(f)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # obj['mated'] = [e for e in obj['mated'] if e > 0]
    # # obj['mated'] = [e - 0.09 for e in obj['mated'] if e > 0]
    # # obj['non-mated'] = [e - 0.09 for e in obj['non-mated'] if e > 0]
    # print(len(obj['mated']))
    # print(np.average(obj['mated']), np.average(obj['non-mated']))
    # ax1.set_title('Voxceleb Noise distance') # dataset name
    # ax1.set_xlabel("hamming distance")
    # sns.kdeplot(data=obj,  fill=True, common_norm = False, )
    # fig.savefig('../dumps/voxceleb_dist_2.png') #dataset specific


    P = '../dumps/lfw_mean_p2' # datset json path
    import json
    with open(P, 'r') as f:
        obj = json.load(f)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    print(np.average(obj['coorelation']))
    # ax1.set_title('NgNet(CFP) Mean distance') # dataset name
    # ax1.set_xlabel("cosine distance (coorelation)")
    # sns.kdeplot(data=obj,  fill=True, common_norm = False)
    # fig.savefig('../dumps/cfp_mean_p2.png') #dataset specific