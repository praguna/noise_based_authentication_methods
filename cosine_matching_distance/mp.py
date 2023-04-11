'''
Call the parties in parallel
'''
import shlex, subprocess

def helper_subprocess():
    def helper(cmd):
        return subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE).communicate()
    return helper

H = helper_subprocess() # use this closure for simplicity

# # start the sockets in the background
# subprocess.Popen(shlex.split('python party_2.py 3001 &'))
# # kill the parallel tcp processes for the server
# out, err = H('lsof -t -i:3001')
# assert err == None
# out, err = H(f'kill -9 {int(out)}')
# # assert err == None