import numpy as np
from one_parameter_defense import one_parameter_defense
import seaborn as sns

def cbe_random(d):
    r = np.random.randn(d)
    bernouli = r.copy()
    bernouli[r > 0] = 1
    bernouli[r <= 0] = -1
    return [r, bernouli]

def cbe_prediction(model: list, X : np.ndarray):
    r, bernouli = model
    A = X.copy()
    A = A * bernouli
    fft_r = np.transpose(np.fft.fft(r).conj())
    fft_x = np.fft.fft(A)
    fft_b = fft_x * fft_r
    B_time =  np.fft.ifft(fft_b)
    B_time = np.real(B_time)
    B = np.zeros(B_time.shape)
    B[B_time>=0] = 1
    B[B_time<0] = 0
    return B

def cbe_prediction_with_opd(model : list, X : np.ndarray):
    X_d = one_parameter_defense(X)
    return cbe_prediction(model, X_d)


if __name__ == "__main__":
    model = cbe_random(512)
    V = []
    import tqdm
    for i in tqdm.tqdm(range(1000)):
        A = np.random.randn(512)
        A /= np.linalg.norm(A)
        B1 = one_parameter_defense(A)
        B2 = one_parameter_defense(A)
        # print(np.random.randn(512, 3000).shape)
        v1 = cbe_prediction(model, B1)
        v2 = cbe_prediction(model, B2)
        
        hd = np.sum(np.logical_xor(v1, v2)) / A.shape[0]
        V.append(hd)
    print(np.average(V) * 100, np.max(V) * 100, np.min(V) * 100)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Perturbation & CBE-rand') # dataset name
    ax1.set_xlabel("hamming distance")
    sns.kdeplot(data=V,  fill=True, common_norm = False)
    plt.show()