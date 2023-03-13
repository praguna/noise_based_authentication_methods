import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1
from torchvision.models.feature_extraction import create_feature_extractor

device  = 'cuda' if torch.cuda.is_available() else 'cpu'

# network
class Nvgnet(nn.Module):
    def __init__(self):
         super(Nvgnet, self).__init__()
         self.arch = InceptionResnetV1()
         self.backend_layer = create_feature_extractor(self.arch, return_nodes={'avgpool_1a' : 'back_out0'})
         self.summaryVec = nn.Sequential(
             nn.Linear(1792, 512, bias=False),
             nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
         )
         self.nonLinearOut = nn.Sequential(
             nn.Linear(512, 128),
             nn.LeakyReLU()
         )

    def forward(self, x):
        x = self.backend_layer(x)['back_out0']
        x = self.summaryVec(x)
        x = self.nonLinearOut(x)
        return x

if __name__ == "__main__":
    nvgNet = Nvgnet().to(device)
    in0 = torch.randn([1, 3, 256, 256]).to(device)
    print(nvgNet(in0).shape)
    