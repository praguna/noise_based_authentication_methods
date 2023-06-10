"""
Server Side, operating using socket
"""
from binary_cosine_local import *
from BNAuth import *
import socket, json, tqdm
from sys import argv
import logging

config = logging.basicConfig(filename='P1.log', level=logging.INFO)

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

if __name__ == "__main__":
    for _ in range(1): 
     try:
          client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
          client.connect(('0.0.0.0', int(argv[1]))) 
          # client.settimeout(100)
          client.send(bytes('-', encoding="utf-8"))

          # N = np.array([np.random.randint(0, 2) for _ in range(200)])
          N = np.zeros((200,))
          N[0 : 30] = 1
          X = get_Random_X(64, False)
          #hard-coding noise
          call_back = bin_2_float_call_back(i_size * 2 , d_size * 2) # to get the float answer
          # call_back = None
          noise = None
          bNAuth = BNAuth(X, N, R = R, party_type = Party.P1, socket = client, call_back = call_back)
          if call_back is None: noise = bNAuth.load(int(argv[1])) #replace with other values
          s = time()
          d1 = bNAuth.perform_secure_match(size=t_size, noise=noise)
          e = time()
          if not call_back:
              logging.info(f'{d1.astype(np.int8)} | {int(argv[1])} | {e-s}') # return to console 
              client.close() # comment this for 1 process computation
              continue
          if call_back:
               print(e - s, ' seconds')
          ## second
          X = get_Random_X(5, True)
          bNAuth1 = BNAuth(X, N, R = R, party_type = Party.P1, socket = client, call_back = call_back)
          bNAuth1.octets, bNAuth1.selected_octect_index = bNAuth.octets, bNAuth.selected_octect_index # sanity check
          d2 = bNAuth1.perform_secure_match(size=t_size)
          d_x = abs(d1) + abs(d2)
          if d_x == abs(d1) : print('res : ',d1)
          else: print('invalid octet set : ', d1)
          e = time()
          print(e - s, ' seconds')
     except Exception as e: 
         raise e
         print(f'error at port : {argv[1]}', e)
     finally:  client.close()
