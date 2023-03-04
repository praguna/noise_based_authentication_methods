import numpy as np
from one_parameter_defense import one_parameter_defense

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


if __name__ == "__main__":
    model = cbe_random(512)
    V = []
    for i in range(100):
        A = np.random.randn(512)
        A /= np.linalg.norm(A)
        B = one_parameter_defense(A)
        # print(np.random.randn(512, 3000).shape)
        v1 = cbe_prediction(model, A)
        v2 = cbe_prediction(model, B)
        hd = 1 - np.sum(np.logical_xor(v1, v2)) / A.shape[0]
        V.append(abs(A @ B - hd))
    print(np.average(V) * 100, np.max(V) * 100, np.min(V) * 100)
    import matplotlib.pyplot as plt
    plt.hist(np.round(V, 3))
    plt.show()