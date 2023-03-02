import numpy as np

def one_parameter_defense(A, m = 5, e = 0.0):
    # sort the array
    s_i = np.argsort(A)
    SA = A[s_i]
    # maximum difference between any 2 adjacent values
    # will be <1 for m > 4, hence sensitivity is 1
    delta_u = 1
    low = -1
    n = len(A)
    # compute sub_range tuples
    for i in range(n):
        next_low = (SA[i] + SA[i+1]) / 2 if i < n-1 else 1
        # discrete subranges
        discrete_vals = np.linspace(low, next_low, m, False)
        # compute utility scores
        utility_prob = np.exp(-np.abs(SA[i] - discrete_vals) * e / (2 * delta_u))
        utility_prob /= np.sum(utility_prob)
        SA_prime = np.random.choice(a = discrete_vals, p = utility_prob)
        SA[i] = SA_prime
        # select at random
        low = next_low
    SA_unsorted = A.copy()
    SA_unsorted[s_i] = SA
    # return the normalized vector with replaced values, unsort the array
    SA_unsorted = SA_unsorted / np.linalg.norm(SA_unsorted)
    return SA_unsorted

if __name__ == '__main__':
    p = []
    for i in range(100):
        # print(np.linspace(-1, 1, 5))
        A = np.random.randn(512)
        # A = np.array([0.2, 0.8, -0.3])
        A = A / np.linalg.norm(A)
        # obfuscated output
        EA1 = one_parameter_defense(A)
        EA2 = one_parameter_defense(A)
        p.append(EA1 @ EA2)
    print(np.average(p), np.max(p), np.min(p))
    import matplotlib.pyplot as plt
    plt.hist(np.round(p, 2))
    plt.show()


# x + N(0, var) + W(x)
# (x + N(0, var)) * W(x) = x * W(x) + N(0, var) * W(x)