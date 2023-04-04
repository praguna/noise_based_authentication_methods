"""
Server Side, operating using socket
"""
from binary_cosine_local import *
from BNAuth import *
import socket, json, tqdm
from sys import argv
from mp import *
import os

key_default = 'key_5400'

with open('random_client_key.json', 'r') as f:
     k = json.load(f)[key_default]
     R = np.array([int(e) for e in k])

i_size , d_size = 1, 8
t_size = i_size + d_size

def get_Random_X(n = 512, zeros = False):
     # example array on b.s
     X = np.random.randn(n)
     X = X / np.linalg.norm(X, 2)
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
     return X

def runcode(p):
    print(f'authenticating at a secure parallel process at port : {p}')
    subprocess.Popen(shlex.split(f'python party_1.py {str(p)} &'))

if __name__ == "__main__":
    start = 'INFO:root:['
    for _ in tqdm.tqdm(range(5)): 
     try:
        subprocess.Popen(shlex.split(f'rm P1.log'))
        P = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007]
        for p in P: runcode(p)
        s = time()
        call_back = bin_2_float_call_back(i_size * 2 , d_size * 2) # to get the float answer
        while True: # await for processing to complete
            if os.path.exists("P1.log"):
                with open("P1.log", 'r') as f:
                    lines = f.readlines()
                    if len(lines) == len(P):
                        port_X = []
                        for line in lines:
                            x = np.fromstring(line[len(start) : line.index(']')], dtype=np.int8, sep=' ')
                            p = int(line.split('|')[1])
                            port_X.append((x, p))
                        port_X.sort(key = lambda e : e[1])
                        # print('processing complete !', port_X)
                        X = [e[0] for e in port_X]
                        e = time()
                        # print(e - s)
                        break
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('0.0.0.0', int(argv[1]))) 
        client.send(bytes('-', encoding="utf-8"))
        N = np.zeros((200,))
        N[0 : 30] = 1
        bNAuth = BNAuth(np.zeros(100), N, R = R, party_type = Party.P1, socket = client, call_back = call_back)
        bNAuth.octets = np.array([[0,0,0,1]]) # replace with original later
        bNAuth.selected_octets = np.array([[0,0,0,1]])
        bNAuth.selected_octect_index = np.array([0])
        d = bNAuth.perform_secure_match_parallel_inputs(X, t_size)
        e = time()
        print(e - s)
        print(d)
        
          
     except Exception as e: 
         print('Errorin network : ', e, 'dropping this')
     finally:  client.close()
