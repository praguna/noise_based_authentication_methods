from time import time
from utils import Party, M as AND_MATRIX, convert_to_str, xor_on_numpy1D, xor_on_numpy2D, serialize_nd_array
import numpy as np
import socket, json, tqdm
from collections import deque
import struct, gzip, pickle as pk

# setting the seed 
# np.random.seed(42)
oc1 , oc2 =  np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
class BNAuth(object):
    '''
    Implementation of Biometric matching with noise without exchanging templates
    '''

    def __init__(self, X, N, party_type : Party, R = [], socket: socket.socket = None, direct_index = False ,error_rate = 0.25, call_back = None) -> None:
        '''
        arg X -> template
        arg MI -> masked indices
        arg party_addr -> other party's addr
        '''
        self.X = X.astype(np.int8)
        if len(R) > 0: self.R = R.astype(np.int8)
        self.N = N.astype(np.int8)
        self.d = len(self.X)
        self.error_rate = error_rate
        self.m = min(len(self.N) // 4, 100)
        self.party_type = party_type 
        self.socket = socket
        self.message_queue = deque()
        self.n = int(0.90 * self.m)
        self.num_dist = 1
        self.selected_octect_index = []
        self.T = []
        self.visited_mask_pos = np.zeros_like(self.N)
        self.count  = 0
        self.buffer = b''
        self.end = b'END'
        self.call_back = call_back
        self.oc_count = -1
        self.octets = []
        self.X1 = None
        self.Y1 = None
        self.R1 = None
        assert self.m > 0
    

    def send_to_peer(self, data):
        '''
        sends it to the peer socket
        '''
        self.socket.sendall(bytes(data, encoding="utf-8") + self.end)
        self.count+=len(data)
        # print(f'sent : {data}')

    def process_bytes(self, decompress_func = None):
        '''
        return the json object recieved
        '''
        b : bytes = self.message_queue.popleft()
        if decompress_func : 
            self.count+=len(b)
            msg = str(decompress_func(b), 'utf-8')
        else:
            msg = b.decode('utf-8')
            self.count += len(msg)
        return json.loads(msg)


    
    def recieve_from_peer(self, decompress_func = None, num_bytes = 16000):
        '''
        decode and extract data from the peer
        '''
        if len(self.message_queue) > 0:
            # print(self.message_queue)
            return self.process_bytes(decompress_func)
        
        b = self.socket.recv(num_bytes)

        while not b.endswith(self.end):
            b+=self.socket.recv(num_bytes)
        
        # print('recieved : ' , b)
            
        buf_split = b.split(self.end)

        self.message_queue.extend(buf_split)
        
        self.message_queue.pop() # empty buffer

        return self.process_bytes(decompress_func)


    def recieve_has_expected_noise(self, pos : np.ndarray, masked_xor)->bool:
        '''
        P2 sends True if expected noise is present and verified
        '''
        self.send_to_peer(json.dumps({'pos' : serialize_nd_array(pos), 'masked_xor' : int(masked_xor)}))
        has_expected_noise = self.recieve_from_peer()
        return has_expected_noise['has_expected_noise']

    
    '''
    Assuming uncertain position MI the same for a person
    '''
    def get_masked_bit_quad(self):
        '''
        receive / send corresponding 4 masked bit positions, compare xor difference
        '''
        # Note : for testing just append pos beside the result
        def P1():
            is_expected_noise = False
            while not is_expected_noise:
                pos = np.random.choice(range(len(self.N)), 4, False) 
                is_expected_noise  = self.recieve_has_expected_noise(pos, xor_on_numpy1D(self.N[pos]))
            return self.N[pos] # implementing only for p1
        
        def P2():
            s = time()
            is_expected_noise = False
            while not is_expected_noise:
                data = self.recieve_from_peer()
                if xor_on_numpy1D(self.N[data['pos']]) != data['masked_xor'] and np.sum(self.visited_mask_pos[data['pos']]) < 2: is_expected_noise = True
                if is_expected_noise: self.visited_mask_pos[data['pos']] = 1
                self.send_to_peer(json.dumps({'has_expected_noise' :  is_expected_noise}))
                t = time()
                if t - s > 3 : raise Exception("Timing out after 3 secs")
            return self.N[data['pos']] # implementing only for p1

        return P1() if self.party_type == Party.P1 else P2()


    def preprocess(self):
        '''
        creates m octects in the party
        '''
        return np.array([self.get_masked_bit_quad() for _ in range(self.m)])
        

    def create_distributed_vectors(self, X, name = 'Y1'):
        '''
        create 2 vectors as : X = X1 xor X2 and distribute X2 over to party_addr
        '''
        X1 = np.array([np.random.randint(0, 2) for _ in range(len(X))])
        X2 = xor_on_numpy2D([X1, X])
        self.send_to_peer(json.dumps({name : serialize_nd_array(X2)}))
        return (X1, X2)


    def fetch_distributed_vector(self):
        '''
        fetch vector Y1 from party_addr
        '''
        return self.recieve_from_peer()['Y1']


    def send_xors_over(self, x1_xor_a1, y1_xor_b1):
        '''
        sends x1 xor a1 and y1 xor b1 to another party
        '''
        self.send_to_peer(json.dumps({'xors' : (bool(x1_xor_a1), bool(y1_xor_b1))}))

    def recieve_xors(self):
        '''
        recieve xors from the other party i.e :  x2 xor a2 and y2 xor b2
        '''
        a = self.recieve_from_peer()
        return a['xors']

    def compute_Z(self, XA, YB, x, y, c) -> np.ndarray:
        '''
        computes z1 or z2 based on the party type
        '''
        if self.party_type == Party.P1: 
            return  (XA & YB) ^ (x & YB) ^ (y & XA) ^ c
        return (x & YB) ^ (y & XA) ^ c

    def fetch_octect(self):
        '''
        gets an octect for each party
        '''
        def P1():
            idx = np.random.choice(list(self.selected_octect_index), 1)
            self.send_to_peer(json.dumps({'idx' : serialize_nd_array(idx)}))
            return idx
        def P2():
            idx = self.recieve_from_peer()['idx']
            return idx
        idx = P1() if self.party_type == Party.P1 else P2()
        octect = self.octets[idx]
        return octect.ravel()
    
    def fetch_octect_bulk(self, batch_size = 128):
        '''
        gets octects for each party
        '''
        def P1():
            idx = np.random.choice(list(self.selected_octect_index), batch_size)
            msg = bytes(json.dumps({'idx' : serialize_nd_array(idx)}),'utf-8')
            com_msg = gzip.compress(msg) + self.end
            self.count+=len(com_msg)
            self.socket.sendall(com_msg)
            self.socket.recvmsg(1)[0]
            return idx
        def P2():
            idx = self.recieve_from_peer(decompress_func=gzip.decompress)['idx']
            self.socket.sendall(struct.pack('?', True))
            return idx
        idx = P1() if self.party_type == Party.P1 else P2()
        return idx


    def perform_computation_phase_v2(self, octect, w, v, noise = False):
        '''
        optimized without util function calls
        '''
        a1, b1, c1 = octect[2] ^ octect[3] , octect[1] ^ octect[3] , octect[3] if not noise else  octect[0] ^ octect[1] ^ octect[2]
        x1_xor_a1, y1_xor_b1 = a1^w , b1^v
        self.send_xors_over(x1_xor_a1, y1_xor_b1)
        x2_xor_a2, y2_xor_b2  = self.recieve_xors()
        XA = x1_xor_a1 ^ x2_xor_a2
        YB = y1_xor_b1 ^ y2_xor_b2
        Z =  (XA & YB) ^ (w & YB) ^ (v & XA) ^ c1  if self.party_type == Party.P1 else (w & YB) ^ (v & XA) ^ c1 
        return Z

    def perform_computation_phase_v3(self, octect, w, v, noise = False):
        '''
        optimized without util function calls
        '''
        a1, b1, c1 = octect[2] ^ octect[3] , octect[1] ^ octect[3] , octect[3] if not noise else  octect[0] ^ octect[1] ^ octect[2]
        x1_xor_a1, y1_xor_b1 = a1^w , b1^v
        self.socket.sendall(struct.pack('??',x1_xor_a1, y1_xor_b1))
        s = self.socket.recvmsg(2)[0]
        x2_xor_a2, y2_xor_b2  = struct.unpack('??',s)
        self.count+=2
        XA = x1_xor_a1 ^ x2_xor_a2
        YB = y1_xor_b1 ^ y2_xor_b2
        Z =  (XA & YB) ^ (w & YB) ^ (v & XA) ^ c1  if self.party_type == Party.P1 else (w & YB) ^ (v & XA) ^ c1 
        return Z

    
    def perform_computation_phase_v4(self, octects, pairs, noise = False):
        '''
        combined computation phase for independent computation
        '''
        def get_triplet(octect):
            a1, b1, c1 = octect[2] ^ octect[3] , octect[1] ^ octect[3] , octect[3] if not noise else  octect[0] ^ octect[1] ^ octect[2]
            return [a1, b1, c1]

        def get_xors_to_send(a1, b1, p):
            return [a1^p[0] , b1^p[1]]

        def get_Z(P, Q):
            x2_xor_a2, y2_xor_b2 = P
            w, v, c1, x1_xor_a1, y1_xor_b1 = Q
            XA = x1_xor_a1 ^ x2_xor_a2
            YB = y1_xor_b1 ^ y2_xor_b2
            Z =  (XA & YB) ^ (w & YB) ^ (v & XA) ^ c1  if self.party_type == Party.P1 else (w & YB) ^ (v & XA) ^ c1 
            return Z

        A, P = [], []
        for o, p in zip(octects, pairs):
            a1, b1, c1 = get_triplet(o)
            x1, x2 = get_xors_to_send(a1, b1, p)
            A.append((p[0], p[1], c1, x1, x2))
            P.extend([x1, x2])
        num = '?'*len(P)
        self.socket.sendall(struct.pack(num, *P))
        s = self.socket.recvmsg(len(P))[0]
        R = struct.unpack(num,s)
        return [get_Z((R[i], R[i+1]) , A[i // 2]) for i in range(0, len(R), 2)]
    
    def calculate_sum(self, W1, W2):
        '''
        perform complete sum as :
        W = (w11 xor w12).......()
        '''
        bits = xor_on_numpy2D([W1, W2])
        return int(convert_to_str(bits), 2)


    def perform_distillation(self, X1, Y1):
        '''
        get the final octect after elimination
        '''
        assert len(self.octets) >= 0
        if len(self.octets) == 1 : return self.octets[0]


        def P1():
            octect_set = {i : e  for i, e in enumerate(self.octets)}
            for _ in range(self.n):
                if len(octect_set) < 2: break
                # send / fetch indices
                j, match = 0 , True
                indices = np.random.choice(list(octect_set.keys()), 2, False)
                self.send_to_peer(json.dumps({'indices' : serialize_nd_array(indices)}))
                idx = np.random.choice(self.d, self.num_dist, False)
                while j < self.num_dist and match:
                    # ind = np.random.choice(self.d, 1)
                    ind = [idx[j]]
                    x, y = X1[ind], Y1[ind]
                    self.send_to_peer(json.dumps({'pos' : serialize_nd_array(ind)}))
                    A = self.perform_computation_phase_v2(octect_set[indices[0]],x, y)
                    B = self.perform_computation_phase_v2(octect_set[indices[-1]],x, y)
                    z1 = A ^ B
                    # fetch/send z2 / z1 from the other party
                    self.send_to_peer(json.dumps({'z2' : serialize_nd_array(z1)}))
                    z2 = self.recieve_from_peer()['z2']
                    match = z1 == z2
                    j+=1
                if match: # compare z1 and z2
                    del_idx = np.random.choice(indices, 1)
                    self.send_to_peer(json.dumps({'del_idx' : serialize_nd_array(del_idx)}))
                    octect_set.pop(del_idx[0])
                else:
                    for e in indices: octect_set.pop(e)
            return sorted(list(octect_set.keys()))
        
        def P2():
            octect_set = {i : e  for i, e in enumerate(self.octets)}
            for _ in range(self.n):
                if len(octect_set) < 2: break
                j, match = 0 , True
                indices = self.recieve_from_peer()['indices']
                while j < self.num_dist and match:
                    ind = self.recieve_from_peer()['pos']
                    x, y = X1[ind], Y1[ind]
                    A = self.perform_computation_phase_v2(octect_set[indices[0]], x, y)
                    B = self.perform_computation_phase_v2(octect_set[indices[-1]], x, y)
                    z1 = A ^ B
                    # fetch/send z2 / z1 from the other party
                    self.send_to_peer(json.dumps({'z2' : serialize_nd_array(z1)}))
                    z2 = self.recieve_from_peer()['z2']
                    match = z1 == z2
                    j+=1
                if match: # compare z1 and z2
                    del_idx = self.recieve_from_peer()['del_idx']
                    octect_set.pop(del_idx[0])
                else:
                    for e in indices: octect_set.pop(e)
            return sorted(list(octect_set.keys()))
        
        return P1() if self.party_type == Party.P1 else P2()
    
    def secure_not(self, inp):
        '''perform secure not'''
        if self.party_type == Party.P1: return np.logical_not(inp)
        return inp

    def increase_oc_count(self, inc):
        self.oc_count += inc
        if self.oc_count > len(self.selected_octets): 
            self.selected_octets = self.octets[self.fetch_octect_bulk(batch_size=50000)]
            self.oc_count = inc - 1

    ###################################### Cosine Distance Utilities #########################################

    def add_2_numbers(self, b1, b2, size = 16, noise = False):
        '''add 2 binary numbers'''
        c = 0
        assert len(b1) == len(b2)
        R = []
        for i in range(len(b1)-1, -1, -1):
            s = b1[i] ^  b2[i] ^  c
            R.insert(0, s)
            self.increase_oc_count(4)
            octets = [self.selected_octets[self.oc_count-i] for i in range(3)]
            pairs = [(b2[i], b1[i]), (b2[i], c), (b1[i], c)]
            a1, a2, a3 = self.perform_computation_phase_v4(octets, pairs, noise)
            xor = a2 ^ a3
            c_1, c_2 = self.secure_not(a1) , self.secure_not(xor)
            c_and = self.perform_computation_phase_v3(self.selected_octets[self.oc_count-4], c_1, c_2, noise)
            c = self.secure_not(c_and)
        return np.array(R[:size], dtype=np.int8)
    
    def multiply_2_numbers(self, b1, b2, size = 16, R = [], noise = False):
        '''multiply 2 signed numbers'''
        
        def partial_sum(A , e, idx = 0):
            L = []
            r2 = R[size - idx - 1] # index from the end
            octets, pairs = [], []
            for i,x in enumerate(A):
                r1 = R[i]
                self.increase_oc_count(4)
                octets.extend([self.selected_octets[self.oc_count-i] for i in range(4)])
                pairs.extend([(x, e), (x, r2), (e, r1), (r1, r2)])
                
            xors = self.perform_computation_phase_v4(octets, pairs, noise)
            for i in range(0, len(xors), 4):
                q = xors[i] ^ xors[i+1] ^ xors[i+2] ^ xors[i+3]
                L.append(q) # secure AND on encrypted text

            for _ in range(idx): L.append(0) # position extend
            for _ in range(2*size - len(L)): L.insert(0, L[0]) #sign extend
            return L
        
        def add_1(A):
            S, c = [], int(self.party_type == Party.P1)
            for x in reversed(A): 
                s = c ^ x
                S.insert(0, s)
                self.increase_oc_count(1)
                c = self.perform_computation_phase_v3(self.selected_octets[self.oc_count], c, x, noise)
            return S
        
        partial_sums = []
        for i, e in enumerate(reversed(b2)):
            p_s = partial_sum(b1, e, i)
            if i == len(b1)-1: # 2s complement for the last element
                p_s = self.secure_not(p_s) #1s complement
                p_s = add_1(p_s)  # 2s complement
            # assert len(p_s) == (2*size), f'created size : {len(p_s)}, {i}'
            partial_sums.append(np.array(p_s, dtype=np.int8))
        
        final_sum = partial_sums[0]
        for i in range(1, len(partial_sums)):
            final_sum = self.add_2_numbers(final_sum, partial_sums[i], size * 2, noise)

        return final_sum
    
    def cosine_distance(self, size = 16, noise = False):
        '''computes sigma(a[i] * b[i])'''
        A1, A2, R = self.X1, self.Y1 , self.R1
        assert len(A1) == len(A2) and len(A1) <= len(R), f"Lengths of inputs don't match {len(A1), len(A2), len(R)}"
        products = []
        r = []
        for i in range(0, len(A1), size):
            l, r = i , i + size
            e1, e2 = A1[l : r] , A2[l : r]
            if len(R) > 0: r = R[l : r]
            prod = self.multiply_2_numbers(e1, e2, size, r, noise)
            products.append(prod)

        final_sum = products[0] 
        for i in range(1, len(products)):
            final_sum = self.add_2_numbers(products[i], final_sum, size * 2, noise)
        
        return final_sum
    
    
    def perform_secure_match(self, size = 16, noise = False):
        '''
        runs secure matching algorithm on both the parties P1 and P2 independently to compute the cosine distance
        '''
        # X => refers to P(X) = X ^ R , (encrypted floating point representation)
        # assert np.linalg.norm(self.X , 2) == 1 has to be true 
        
        # preprocessing octets ignore for now
        if len(self.octets) == 0:
            self.octets = self.preprocess() 

        # distribute and recieve X,Y as X1, Y1
        if self.X1 is None and self.Y1 is None:
            self.X1, _ = self.create_distributed_vectors(self.X)
            self.Y1 =  np.array(self.fetch_distributed_vector())
            if self.party_type == Party.P2: self.Y1, self.X1 = self.X1, self.Y1

        # perform distillation
        while len(self.selected_octect_index) == 0:
            self.selected_octect_index += self.perform_distillation(self.X1, self.Y1)
        
        if self.R1 is None :
            # distribute R as R1 or R2
            if self.party_type == Party.P1: 
                self.R1,_ = self.create_distributed_vectors(self.R, 'R2')
                self.socket.recvmsg(1)[0]
            else: 
                self.R1 = self.recieve_from_peer()['R2']
                self.socket.sendall(struct.pack('?', True))
        
        # select random octets
        self.selected_octets = self.octets[self.fetch_octect_bulk(batch_size=2000)]
        
        # perform cosine distance
        d1 = self.cosine_distance(size, noise)
        if self.call_back is None: return d1
        self.send_to_peer(json.dumps({'d2' : serialize_nd_array(d1)}))
        d2 = self.recieve_from_peer()['d2']
        d = np.logical_xor(d1, d2)
        return self.call_back(d)
    
    def set_distributed_inputs(self, obj):
        '''
        setting these values for downstream tasks
        '''
        self.X1 = obj['X1']
        self.Y1 = obj['Y1']
        self.R1 = obj['R1']
        self.octets = obj['octets']
        self.selected_octect_index = obj['selected_octect_index']
    
    def precompute_octets(self):
        '''
        performs distillation , preprocessing and distribution of vectors
        '''
        if len(self.octets) == 0:
            self.octets = self.preprocess() 

        # distribute and recieve X,Y as X1, Y1
        if self.X1 is None or self.Y1 is None:
            self.X1, _ = self.create_distributed_vectors(self.X)
            self.Y1 =  np.array(self.fetch_distributed_vector())
            if self.party_type == Party.P2: self.Y1, self.X1 = self.X1, self.Y1

        # perform distillation
        while len(self.selected_octect_index) == 0:
            self.selected_octect_index += self.perform_distillation(self.X1, self.Y1)
        
        if self.R1 is None :
            # distribute R as R1 or R2
            if self.party_type == Party.P1: 
                self.R1,_ = self.create_distributed_vectors(self.R, 'R2')
                self.socket.recvmsg(1)[0]
            else: 
                self.R1 = self.recieve_from_peer()['R2']
                self.socket.sendall(struct.pack('?', True))       


    def perform_secure_match_parallel_inputs(self, sums = [], size = 16, noise = False):
        '''
        take x1, x2, x3 .... xN computed partial sums in parallel and add them together 
        '''
        self.selected_octets = self.octets[self.fetch_octect_bulk(batch_size=5000)]
        final_sum = sums[0] 
        for i in range(1, len(sums)):
            final_sum = self.add_2_numbers(sums[i], final_sum, size * 2, noise)

        self.send_to_peer(json.dumps({'d2' : serialize_nd_array(final_sum)}))
        d2 = self.recieve_from_peer()['d2']
        d = np.logical_xor(final_sum, d2)
        return self.call_back(d)

    def save(self, ports = [], noise = False):
        with open(f'{self.party_type}.pk' , 'wb') as f: 
            obj = {'octets' : self.octets, 'selected_octect_index' : self.selected_octect_index, 'noise' : noise}
            batch_size = len(self.X) // len(ports)
            for i, p in enumerate(ports):
                obj[f'{p}_X1'] = self.X1[batch_size * i : batch_size * (i+1)]
                obj[f'{p}_Y1'] = self.Y1[batch_size * i : batch_size * (i+1)]
                obj[f'{p}_R1'] = self.R1[batch_size * i : batch_size * (i+1)]
            pk.dump(obj, f)
    
    def load(self, port = 3000):
        with open(f'{self.party_type}.pk' , 'rb') as f: 
            obj = pk.load(f)
        obj['X1'] = obj[f'{port}_X1']
        obj['Y1'] = obj[f'{port}_Y1']
        obj['R1'] = obj[f'{port}_R1']
        self.set_distributed_inputs(obj)
        return obj['noise']