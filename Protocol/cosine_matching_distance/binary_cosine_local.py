from time import time
import numpy as np
dec_pos, dec_neg = 0.4, -0.8989324


def float_2_complement_decimal(intBits,decBits,number):
    if decBits == 0:
        mx = pow(2,intBits-1) - 1 # maximum number
    else:
        mx = pow(2,intBits-1) - pow(2,-1*decBits) # maximum number
    mn = -1*pow(2,intBits-1) # minimum number
    if number > mx:
        print ("number:" + str(number) + " has been truncated to: " + str(mx))
        number = mx
    elif number < mn:
        print ("number:" + str(number) + " has been truncated to: " + str(mn))
        number = mn
    n = []
    m = 0
    if number < 0:
        n.append(1)
        m = -1*pow(2,intBits-1)
    else:
        n.append(0)
        m = 0
        
    for i in reversed(range(intBits-1)):
        m1 = m + pow(2,i)
        if number < m1:
            n.append(0)
        else:
            n.append(1)
            m = m1
    for i in range(1,decBits+1):
        m1 = m + pow(2,-1*i)
        if number < m1:
            n.append(0)
        else:
            n.append(1)
            m = m1
    return ''.join([str(i) for i in n])

def decimal_2_complement_float(intBits,decBits,binString):
    n = 0.0
    if binString[0] == "1":
        n = -1*pow(2,intBits-1)
    for i in range(intBits-1):
        n += int(binString[intBits-1-i])*pow(2,i)
    for i in range(1,decBits+1):
        n += int(binString[intBits-1+i])*pow(2,-1*i)
    return n

def add_2_numbers(b1, b2, size = 16):
    '''add 2 binary numbers'''
    c = 0
    assert len(b1) == len(b2)
    R = []
    for i in range(len(b1)-1, -1, -1):
        s = b1[i] ^  b2[i] ^  c
        R.insert(0, s)
        a1, a2, a3 = b2[i] and b1[i], b2[i] and c, b1[i] and c
        xor = a2 ^ a3
        # c =  not(not(a1) and not(xor))
        c = a1 or xor
    return np.array(R[:size], dtype=np.int8)


def multiply_2_numbers(b1, b2, size = 16, R = []):
    '''multiply 2 signed numbers'''
    
    def partial_sum(A , e, idx = 0):
        L = []
        r1, r2 = None , None
        if len(R) > 0 : r2 = R[size - idx - 1] # index from the end
        for i, x in enumerate(A): 
            if len(R) > 0: r1 = R[i] #index from begining
            if r1 is None or r2 is None: L.append(x and e) #secure AND on plain text
            else:
                m, n, o, p = x and e, x and r2, e and r1, r1 and r2
                q = m ^ n ^ o ^ p
                L.append(q)  # secure AND on encrypted text
        for _ in range(idx): L.append(0) # position extend
        for _ in range(2*size - len(L)): L.insert(0, L[0]) #sign extend
        return L
    
    def add_1(A):
        S, c = [], 1
        for x in reversed(A): 
            s = c ^ x
            S.insert(0, s)
            c = c and x
        return S
    
    partial_sums = []
    for i, e in enumerate(reversed(b2)):
        p_s = partial_sum(b1, e, i)
        if i == len(b1)-1: # 2s complement for the last element
            p_s = np.logical_not(p_s) #1s complement
            p_s = add_1(p_s)  # 2s complement
        assert len(p_s) == (2*size), f'created size : {len(p_s)}, {i}'
        partial_sums.append(np.array(p_s, dtype=np.int8))
    
    
    final_sum = partial_sums[0]
    for i in range(1, len(partial_sums)):
        final_sum = add_2_numbers(final_sum, partial_sums[i], size * 2)

    return final_sum

def cosine_distance(A1, A2, size = 16, R = []):
    '''computes sigma(a[i] * b[i])'''
    products = []
    i, r = 0, []
    for e1, e2 in zip(A1, A2):
         if len(R) > 0: r = R[size * i : size * (i+1)]
         prod = multiply_2_numbers(e1, e2, size, r)
         products.append(prod)
         i+=1

    final_sum = products[0] 
    for i in range(1, len(products)):
        final_sum = add_2_numbers(products[i], final_sum, size * 2)
    
    return final_sum


def bin_2_float_call_back(intBits = 2 , decBits = 16):
    '''return a call back from bin str to float'''
    def call_back(X):
        X1 = X
        if type(X) != str: X1 = [str(int(e)) for e in list(X)]
        return decimal_2_complement_float(intBits, decBits, X1)
    return call_back


if __name__ == "__main__":

    # A_1 = float_2_complement_decimal(2, 16, dec_pos)
    # A_2 = decimal_2_complement_float(2, 16, A_1)
    # print(A_1, A_2)
    # B_1 = float_2_complement_decimal(2, 16, dec_neg)
    # B_2 = decimal_2_complement_float(2, 16, B_1)
    # print(B_1, B_2)
    # A_t = np.array([a for a in A_1], dtype=np.int8)
    # B_t = np.array([a for a in B_1], dtype=np.int8)
    # print(A_t, B_t)


    # S_t = add_2_numbers(A_t, B_t, 18)
    # print(S_t)
    # S_d = decimal_2_complement_float(2, 16, ''.join([str(e) for e in S_t]))
    # print(A_2 + B_2, S_d)
    # assert S_d - (A_2 + B_2) == 0



    # P_t = multiply_2_numbers(A_t, B_t, size=18)
    # P_d = decimal_2_complement_float(4, 32, ''.join([str(e) for e in P_t]))
    # print(P_t, P_d, A_2*B_2)
    # assert P_d == (A_2 * B_2), f'{P_d} , {A_2 * B_2}'



    Arr1 = np.random.randn(512)
    Arr1 = Arr1 / np.linalg.norm(Arr1, 2)
    Arr2 = np.random.randn(512)
    Arr2 = Arr2 / np.linalg.norm(Arr2, 2)
    assert 1.0 - np.linalg.norm(Arr1) < 1e-5
    assert 1.0 - np.linalg.norm(Arr2) < 1e-5
    Arr1_t = []
    Arr2_t = []
    P_d, P_c = 0, 0
    R = np.array([np.random.choice([0, 1]) for _ in range(16 * 512)])
    # R = np.zeros(16, dtype=np.int8)
    i = 0
    i_size , d_size = 2, 8
    t_size = i_size + d_size
    for a,b in zip(Arr1, Arr2):
        x = float_2_complement_decimal(i_size , d_size , a)
        y = float_2_complement_decimal(i_size , d_size, b)
        arr1 = np.logical_xor(np.array([a for a in x], dtype=np.int8), R[i * t_size : t_size * (i+1)])
        arr2 = np.logical_xor(np.array([a for a in y], dtype=np.int8), R[i * t_size : t_size * (i+1)])
        Arr1_t.append(arr1)
        Arr2_t.append(arr2)
        P_c += (a * b)
        i+=1
    print()
    print('Actual Distance' , P_c)
    s = time()
    C_t = cosine_distance(Arr1_t, Arr2_t, size=t_size, R = R)
    e = time()
    print(e - s, 'seconds')
    C_d = decimal_2_complement_float(i_size * 2 , d_size * 2, ''.join([str(e) for e in C_t]))
    print('Calculated : ',C_t, C_d)
    assert abs(P_c - C_d) < 1e-2