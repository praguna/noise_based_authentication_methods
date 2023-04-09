from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    P = '../dumps/celeb_a_dist' # datset json path
    import json
    with open(P, 'r') as f:
        obj = json.load(f)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Celeb A Noise distance') # dataset name
    ax1.set_xlabel("hamming distance")
    sns.kdeplot(data=obj,  fill=True, common_norm = False)
    fig.savefig('../dumps/vis_data_celeb_A.png') #dataset specific