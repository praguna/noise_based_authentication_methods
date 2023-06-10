"""
Client Side, operating using socket
"""
from binary_cosine_local import *
from BNAuth import *
import socket, json
from sys import argv
from mp import *


key_default = 'key_9600'

with open('random_client_key.json', 'r') as f:
     k = json.load(f)[key_default]
     R = np.array([int(e) for e in k])


i_size , d_size = 1, 8
t_size = i_size + d_size

def get_Random_X(n = 512, y = None):
     # example array on b.s
     X = np.ones(n)
     X = X / np.linalg.norm(X, 2)
    #  # print(X)
    #  X = np.ones(n) /  np.linalg.norm(np.ones(n), 2)
     if y is not None: X = y
     # R = np.zeros((200, ))
    #  X = np.zeros(n)
     Arr1_t = []
     i = 0
     for a in X:
          x = float_2_complement_decimal(i_size , d_size, a)
          arr1 = list(np.logical_xor(np.array([a for a in x], dtype=np.int8), R[i * t_size : t_size * (i+1)]))
          Arr1_t.extend(arr1)
          i+=1

     X = np.array(Arr1_t).astype(np.int8)
     return X

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
    serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serv.bind(('0.0.0.0', int(argv[1])))
    serv.listen()
    print(f'server started at {argv[1]}!!!')
    subprocess.Popen(shlex.split(f'rm P2.log'))
    P = list(range(3000, 3000 + 16)) #has to be ordered
    # start multiple processes
    for p in P: teardown(p)
    for p in P: startup(p)
    call_back = bin_2_float_call_back(i_size * 2 , d_size * 2) # to get the float answer
    while True:
       try:
            # accept connection 
            I = get_Random_X(32)
            conn, addr = serv.accept()
            mode = conn.recv(9600 * 4).decode('utf-8')
            conn.send(bytes('-', encoding="utf-8"))
            N = np.zeros(512)
            m, y = mode.split(',')
            y = np.array([float(e.strip()) for e in y[1:-1].split(' ') if len(e) > 0])
            I = get_Random_X(32, y)
            # N = np.array([np.random.choice([0,1]) for _ in range(512)])
            # print(np.sum(np.logical_xor(N , np.array([int(e) for e in mode.split(',')]))) / len(N))
            # N[np.random.choice(len(N), 256)] = 1
            # bNAuth = BNAuth(I, N , party_type = Party.P2, socket = conn, call_back = call_back)
            # # bNAuth.precompute_octets()
            # # error correction for a zero vector
            # noise = int(m) == 1
            # for _ in range(1):
            #     bNAuth.selected_octect_index = []
            #     r, check = bNAuth.perform_secure_match(size = t_size, noise = noise)
            #     # print(check)
            # print(bNAuth.selected_octect_index)
            # exit(0)
            ## d = bNAuth.perform_secure_match(size=t_size, noise = noise) #one last time
            # continue
            ## bNAuth.X = get_Random_X(512)
            ## bNAuth.X1 = None #distribute inputs
            noise = int(m) == 1
            bNAuth = BNAuth(I, N , party_type = Party.P2, socket = conn, call_back = call_back)
            bNAuth.precompute_octets()
            # continue
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
            PX = [e[0] for e in port_X]
            # bNAuth = BNAuth(np.zeros(100), np.zeros(200), party_type = Party.P2, socket = conn, call_back = call_back)
            # bNAuth.octets = np.array([[0,0,0,0]]) # replace with original later
            # bNAuth.selected_octets = np.array([[0,0,0,0]])
            # bNAuth.selected_octect_index = [0]
            d, check = bNAuth.perform_secure_match_parallel_inputs(PX, t_size, noise)
            # print(bNAuth.count)
        #  print(d)
         
        
       except Exception as e: 
        #    raise e
           print('Error mp : ',e)
           for p in P: teardown(p)
           for p in P: startup(p)
       finally: conn.close()