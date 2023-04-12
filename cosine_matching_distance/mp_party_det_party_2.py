"""
Client Side, operating using socket
"""
from binary_cosine_local import *
from BNAuth import *
import socket, json
from sys import argv
from mp import *


### Dataset / Modality specific import
import mp_party_1_soco as mp_det
#####

key_default = 'key_5400'

with open('random_client_key.json', 'r') as f:
     k = json.load(f)[key_default]
     R = np.array([int(e) for e in k])


i_size , d_size = 1, 8
t_size = i_size + d_size

def get_Random_X(n = 512):
     # example array on b.s
     X = np.random.randn(n)
     X = X / np.linalg.norm(X, 2)
     # print(X)
     X = np.ones(n) /  np.linalg.norm(np.ones(n), 2)

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

def get_X(X):
     Arr1_t = []
     i = 0
     for a in X:
          x = float_2_complement_decimal(i_size , d_size, a)
          arr1 = list(np.logical_xor(np.array([a for a in x], dtype=np.int8), R[i * t_size : t_size * (i+1)]))
          Arr1_t.extend(arr1)
          i+=1

     return np.array(Arr1_t).astype(np.int8)

def startup(p):
    print(f'starting a secure parallel process at port : {p}')
    subprocess.Popen(shlex.split(f'python party_2.py {str(p)} &'))

def teardown(p):
    print(f'tearing a secure parallel process at port : {p}')
    out, err = H(f'lsof -t -i:{str(p)}')
    assert err == None
    if len(out) == 0: return
    out, err = H(f'kill -9 {int(out)}')
    if err: print(err)

if __name__ == "__main__":
    start = 'INFO:root:['
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind(('0.0.0.0', int(argv[1])))
    serv.listen()
    print(f'server started at {argv[1]}!!!')
    # subprocess.Popen(shlex.split(f'rm P2.log'))
    P = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007] #has to be ordered
    # start multiple processes
    for p in P: teardown(p)
    for p in P: startup(p)
    call_back = bin_2_float_call_back(i_size * 2 , d_size * 2) # to get the float answer
    while True:
       try:
            # accept connection 
            I = get_Random_X(32)
            conn, addr = serv.accept()
            mode = conn.recv(8096).decode('utf-8')
            conn.send(bytes('-', encoding="utf-8"))

            # N = np.zeros(200) #base case
            N = mp_det.extract_noise([mode.split(' ')[1]])

            bNAuth = BNAuth(I, N, party_type = Party.P2, socket = conn, call_back = call_back)
            # bNAuth.precompute_octets()
            # error correction for a zero vector
            noise = True
            for _ in range(1):
                bNAuth.selected_octect_index = []
                r = bNAuth.perform_secure_match(size = t_size, noise = False)
                if abs(r - 0.98876953125) < 1e-5:
                    noise = False
                    break
            # d = bNAuth.perform_secure_match(size=t_size, noise = noise) #one last time
            # P = mp_det.extract_noise([mode.split(' ')[0]])
            # bNAuth.X = get_Random_X(512) # base
            print(bNAuth.octets[bNAuth.selected_octect_index], noise)
            bNAuth.X = get_X(mp_det.extract_feature([mode.split(' ')[0]]).flatten())
            bNAuth.X1 = None #distribute inputs
            bNAuth.precompute_octets()
            # continue
            # print(noise)
            bNAuth.save(P, noise)
            # continue
            p = conn.recv(8096).decode('utf-8')
            if p != 'M': raise Exception('close the client and tear down')
            with open("P2.log", 'r') as f: lines = f.readlines()
            port_X = []
            for line in lines[-len(P):]:
                x = np.fromstring(line[len(start) : line.index(']')], dtype=np.int8, sep=' ')
                p = int(line.split('|')[1])
                port_X.append((x, p))
            port_X.sort(key = lambda e : e[1])
            X = [e[0] for e in port_X]
            # bNAuth = BNAuth(np.zeros(100), np.zeros(200), party_type = Party.P2, socket = conn, call_back = call_back)
            # bNAuth.octets = np.array([[0,0,0,0]]) # replace with original later
            # bNAuth.selected_octets = np.array([[0,0,0,0]])
            # bNAuth.selected_octect_index = [0]
            d = bNAuth.perform_secure_match_parallel_inputs(X, t_size, noise)
        #  print(d)
         
        
       except Exception as e: 
        #    raise e
           print('Error mp : ',e)
           for p in P: teardown(p)
           for p in P: startup(p)
       finally: conn.close()