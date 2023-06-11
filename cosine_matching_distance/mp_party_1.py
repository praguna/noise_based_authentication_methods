"""
Server Side, operating using socket
"""
from binary_cosine_local import *
from BNAuth import *
import socket, json, tqdm
from sys import argv
from mp import *
import os

key_default = 'key_9600'

with open('random_client_key.json', 'r') as f:
     k = json.load(f)[key_default]
     R = np.array([int(e) for e in k])

i_size , d_size = 1, 8
t_size = i_size + d_size

def get_Random_X(n = 512, zeros = False):
     # example array on b.s
     X1 = np.random.randn(n)
     X1 = X1 / np.linalg.norm(X1, 2)
     X = X1[:]
     # print(X)
     if zeros : X = np.zeros(n)

     # R = np.zeros((200, ))
     Arr1_t = []
     i = 0
     for a in X:
          x = float_2_complement_decimal(i_size , d_size, a)
          arr1 = list(np.logical_xor(np.array([a for a in x], dtype=np.int8), R[i * t_size : t_size * (i+1)]))
          Arr1_t.extend(arr1)
          i+=1

     X = np.array(Arr1_t).astype(np.int8)
     return X, X1

def runcode(p):
    # print(f'authenticating at a secure parallel process at port : {p}')
    subprocess.Popen(shlex.split(f'python party_1.py {str(p)} &'))

if __name__ == "__main__":
    start = 'INFO:root:['
    call_back = bin_2_float_call_back(i_size * 2 , d_size * 2) # to get the float answer
    L = []
    for _ in tqdm.tqdm(range(100)): 
     try:
        subprocess.Popen(shlex.split(f'rm P1.log'))
        N = np.zeros((512,))
        # N = np.array([np.random.choice([0,1]) for _ in range(512)])
        N[0 :  int(0.25 * (len(N)))] = 1
        X, X1 = get_Random_X(512)
        Y, Y1 = get_Random_X(512)
        v = (X1 @ Y1)
        check, noise , i = 1 , False , 0
        while check > 0:
        #     noise = i%2 != 0
        #     client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        #     # client.settimeout(100)
        #     client.connect(('0.0.0.0', int(argv[1]))) 
        #     client.send(bytes(str(int(noise)) + ',' + str(Y1), encoding="utf-8"))
        #     client.recv(8096).decode('utf-8')
        #     bNAuth = BNAuth(X, N, R = R, party_type = Party.P1, socket = client, call_back = call_back)
        #     # bNAuth.precompute_octets()
        #     # noise = True
        #     # error correction
        #     bNAuth.selected_octect_index = []
        #     d, check = bNAuth.perform_secure_match(size=t_size, noise = noise)
        #     print(bNAuth.selected_octect_index)
        #     print(bNAuth.octets[bNAuth.selected_octect_index])
        #     print(d, v, check)
        #     if not noise and check == 0: print('valid')
        #     print(noise , check, i+1)
        #     i+=1
        #     client.close()
        # ## d = bNAuth.perform_secure_match(size=t_size, noise = noise) #one last time
        # L.append(i)
        # continue
            # # bNAuth.X = get_Random_X(512, False)
            # # bNAuth.X1 = None #distribute inputs
            subprocess.Popen(shlex.split(f'rm P1.log'))
            noise = i%2 != 0
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # client.settimeout(100)
            Y1_str = '['+ ''.join([str(e)+' ' for e in Y1])  + ']'
            client.connect(('0.0.0.0', int(argv[1]))) 
            client.send(bytes(str(int(noise)) + ',' + Y1_str, encoding="utf-8"))
            client.recv(8096).decode('utf-8')
            bNAuth = BNAuth(X, N, R = R, party_type = Party.P1, socket = client, call_back = call_back)
            bNAuth.precompute_octets()
            # purity
            # n = 0 
            # for e in bNAuth.octets[bNAuth.selected_octect_index]:
            #     if np.sum(e) == 1: n+=1
            # L.append(max(n / len(bNAuth.octets[bNAuth.selected_octect_index]), 1 - (n / len(bNAuth.octets[bNAuth.selected_octect_index]))))
            # check = 0
            # purity
            # continue
            P = list(range(3000, 3000 + 16)) #has to be ordered
            bNAuth.save(P, noise)
            # continue
            for p in P: runcode(p)
            s = time()
            while True: # await for processing to complete
                if os.path.exists("P1.log"):
                    with open("P1.log", 'r') as f:
                        lines = f.readlines()
                        if time() - s > 3 : raise Exception('time out!!')
                        if len(lines) == len(P):
                            port_X = []
                            for line in lines:
                                x = np.fromstring(line[len(start) : line.index(']')], dtype=np.int8, sep=' ')
                                p = int(line.split('|')[1])
                                port_X.append((x, p))
                            port_X.sort(key = lambda e : e[1])
                            # print('processing complete !', port_X)
                            PX = [e[0] for e in port_X]
                            # e = time()
                            # print(e - s)
                            #indicate processing is complete
                            client.send(bytes('M', encoding="utf-8"))
                            break
            # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # client.connect(('0.0.0.0', int(argv[1]))) 
            # client.send(bytes('-', encoding="utf-8"))
            # N = np.zeros((200,))
            # N[0 : 30] = 1
            # bNAuth = BNAuth(np.zeros(100), N, R = R, party_type = Party.P1, socket = client, call_back = call_back)
            # bNAuth.octets = np.array([[0,0,0,1]]) # replace with original later
            # bNAuth.selected_octets = np.array([[0,0,0,1]])
            # bNAuth.selected_octect_index = np.array([0])
            # bNAuth.save()
            # bNAuth.load()
            d, check = bNAuth.perform_secure_match_parallel_inputs(PX, t_size, noise)
            # print(bNAuth.selected_octect_index)
            # print(bNAuth.octets[bNAuth.selected_octect_index])
            check = int(check)
            if not noise and check == 0: print('valid')
            i+=1
            e = time()
            L.append(e - s)
            # print(bNAuth.count, check)
            if check  == 0:
                assert abs(d - v) < 1e-1, (d, v, bNAuth.octets[bNAuth.selected_octect_index], noise)
            print(noise ,d, v, check, i)
     except Exception as e: 
        #  raise e
         if isinstance(e, AssertionError):
            raise e
         print('Error : ', e, 'dropping this')
     finally:  client.close()
    print(np.average(L), np.max(L), np.min(L))