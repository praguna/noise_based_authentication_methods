"""
Client Side, operating using socket
"""
from binary_cosine_local import *
from BNAuth import *
import socket, json
from sys import argv
import logging

config = logging.basicConfig(filename='P2.log', level=logging.INFO)


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

if __name__ == "__main__":
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind(('0.0.0.0', int(argv[1])))
    serv.listen()
    print(f'server started at {argv[1]}!!!')
    while True:
       try:
          X = get_Random_X(64)
          conn, addr = serv.accept()
          mode = conn.recv(8096).decode('utf-8')
          if mode.find(':') != -1: b = True
          #hard-coding noise
          # call_back = bin_2_float_call_back(i_size * 2 , d_size * 2) # to get the float answer
          call_back = None
          bNAuth = BNAuth(X, np.zeros(200), party_type = Party.P2, socket = conn, call_back = call_back)
          if call_back is None: bNAuth.load(int(argv[1])) #replace with other values
          s = time()
          d = bNAuth.perform_secure_match(size=t_size)
          e = time()
          if not call_back: 
          #     with open('out.txt', 'w+') as f: f.writelines(str(d.astype(np.int8)) + ' ' + argv[1]) 
              logging.info(f'{str(d.astype(np.int8))} | {argv[1]}') 
              continue
          ### second times
          X = get_Random_X(5)
          bNAuth1 = BNAuth(X, np.zeros(200), party_type = Party.P2, socket = conn, call_back = call_back)
          bNAuth1.octets, bNAuth1.selected_octect_index = bNAuth.octets, bNAuth.selected_octect_index # sanity check
          d+=bNAuth1.perform_secure_match(size=t_size)
          # print(d)
       except Exception as e: print('Error : ',e)
       finally: conn.close()
