"""
Server Side, operating using socket
"""
from binary_cosine_local import *
from BNAuth import *
import socket, json, tqdm
from sys import argv

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

if __name__ == "__main__":
    for _ in tqdm.tqdm(range(1)): 
     try:
          client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          client.connect(('0.0.0.0', int(argv[1]))) 
          client.send(bytes('-', encoding="utf-8"))

          # N = np.array([np.random.randint(0, 2) for _ in range(200)])
          N = np.zeros((200,))
          N[0 : 30] = 1
          X = get_Random_X(128, False)
          #hard-coding noise
          bNAuth = BNAuth(X, N, R = R, party_type = Party.P1, socket = client, call_back = bin_2_float_call_back(i_size*2, d_size*2))
          s = time()
          d1 = bNAuth.perform_secure_match(size=t_size)
          e = time()
          print(e - s, ' seconds')
          ## second
          X = get_Random_X(5, True)
          bNAuth1 = BNAuth(X, N, R = R, party_type = Party.P1, socket = client, call_back = bin_2_float_call_back(i_size*2, d_size*2))
          bNAuth1.octets, bNAuth1.selected_octect_index = bNAuth.octets, bNAuth.selected_octect_index # sanity check
          print(bNAuth1.octets)
          d2 = bNAuth1.perform_secure_match(size=t_size)
          d_x = abs(d1) + abs(d2)
          if d_x == abs(d1) : print(d1)
          else: print('invalid octet set : ', d1)
          e = time()
          print(e - s, ' seconds')
     except Exception as e: raise e
     finally:  client.close()
