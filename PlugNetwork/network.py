import torch
from torch import nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor

# coorelation loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = torch.cosine_similarity(y_t, y_prime_t)
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

# network
class NvgnetFace(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NvgnetFace, self).__init__()
        self.args = kwargs['args']
        self.arch = InceptionResnetV1(pretrained='vggface2')
        # Freeze all layers
        for param in self.arch.parameters(): param.requires_grad = False

        # Unfreeze the first convolutional layer
        for param in self.arch.conv2d_1a.parameters(): param.requires_grad = True
        self.backend_layer = create_feature_extractor(self.arch, return_nodes={'avgpool_1a' : 'back_out0'})
        self.summaryVec = nn.Sequential(
            nn.Linear(1792, 512, bias=False),
            nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        )
        self.nonLinearOut = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU()
        )

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.correlationLoss = LogCoshLoss().to(self.args.device)
        # used to extract embedding
        self.arch0 = InceptionResnetV1(pretrained='vggface2').eval().to(self.args.device)
        # optimizer 
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

    

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    def compute_loss(self, features, features_prime):
        logits, labels = self.info_nce_loss(features)
        l1 = self.criterion(logits, labels)
        l2 = self.correlationLoss(features, features_prime)
        return l1 * 0.22 + l2 * 0.78        

    def train(self, train_loader):
        for epoch in range(self.args.epoch):
            for images in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                features = self(images)
                features_prime = self.arch0(images)
                loss = self.compute_loss(features, features_prime)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def forward(self, x):
        x = self.backend_layer(x)['back_out0'].view(-1, 1792)
        x = self.summaryVec(x)
        x = self.nonLinearOut(x)
        return x

if __name__ == "__main__":
    # remove this 
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    import argparse
    parser = argparse.ArgumentParser(description='Nvgnet')
    parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
    parser.add_argument('--device', default='cuda', type=str,
                    help='device type')
    parser.add_argument('--lr', default=0.0003, type=float,
                    help='learning rate')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight_decay')
    parser.add_argument('--epoch', default=1, type=int,
                    help='epoch')
    args = parser.parse_args()
    nvgNetFace = NvgnetFace(args=args).to(device)
    in0 = torch.randn([2, 3, 256, 256]).to(device)
    assert nvgNetFace(in0).shape == torch.Size([2, 128])