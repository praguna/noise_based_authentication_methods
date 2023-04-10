from network import *
from facenet_pytorch import MTCNN
from PIL import Image
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse
parser = argparse.ArgumentParser(description='Nvgnet')
parser.add_argument('--temperature', default=0.07, type=float,
                help='softmax temperature (default: 0.07)')
parser.add_argument('--device', default='cuda', type=str,
                help='device type')
parser.add_argument('--lr', default=0.15, type=float,
                help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                help='weight_decay')
parser.add_argument('--epoch', default=1, type=int,
                help='epoch')
parser.add_argument('--batch_size', default=128, type=int,
                help='batch_size')
parser.add_argument('--delta', default=1e-5, type=float,
                help='delta')
parser.add_argument('--wandb', default=True, type=bool,
                help='wandb')
parser.add_argument('--expname', default=True, type=str,
                help='experiment_8')
args = parser.parse_args()
nvgNetFace = NvgnetFace(args=args).to(device).eval()
file = f'/home2/praguna.manvi/plg_models/model_{args.lr}_{args.batch_size}_copy.pt'
nvgNetFace.load_state_dict(torch.load(file))
mtcnn = MTCNN()


def compute_embedding(P1):
    I = mtcnn(Image.open(P1)).to(device).unsqueeze(0)
    _, B = nvgNetFace(I)
    # C = ARCH0(I).detach().to('cpu').numpy().flatten()
    R = B.detach().to('cpu').numpy().flatten()
    R /= np.linalg.norm(R, 2)
    # print(C.shape, R.shape)
    # print(C @ R)
    # exit(0)
    return R

def compute_embedding_with_distance(P1):
    I = mtcnn(Image.open(P1)).to(device).unsqueeze(0)
    _, B = nvgNetFace(I)
    C = ARCH0(I).detach().to('cpu').numpy().flatten()
    R = B.detach().to('cpu').numpy().flatten()
    R /= np.linalg.norm(R, 2)
    # print(C.shape, R.shape)
    # print(C @ R)
    # exit(0)
    return R, C @ R