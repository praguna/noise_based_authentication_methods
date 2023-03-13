import torch

device  = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# train nvgnet with a different private dataset
def train():

    pass


if __name__ == '__main__':
    train()