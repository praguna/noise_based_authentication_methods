import torch
from torch import nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
import warnings
import numpy as np
import wandb
warnings.filterwarnings("ignore")

# used in training and parameter setting
ARCH = InceptionResnetV1(pretrained='vggface2').to('cuda')
ARCH0 = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')

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
        self.backend_layer = create_feature_extractor(ARCH, return_nodes={'avgpool_1a' : 'back_out0'})
        # Freeze all layers
        for param in self.backend_layer.parameters(): param.requires_grad = False
        # Unfreeze the first convolutional layer
        ## for dp-adam training only
        # for param in self.backend_layer.conv2d_1a.parameters(): param.requires_grad = True
        # setting full connected layer to train
        for param in self.backend_layer.block8.parameters(): param.requires_grad = True
        self.summaryVec = nn.Sequential(
            nn.Linear(1792, 512, bias=False),
            nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        )
        self.nonLinearOut = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.correlationLoss = LogCoshLoss().to(self.args.device)
        # used to extract embedding
        # optimizer 
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
    
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)


    def info_nce_loss(self, features):

        batch_size = len(features) // 2
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # std = 4 * np.sqrt(2 * batch_size - 1) * 0.001
        ## uncomment for DP-ADAM / DP-SGD training
        # dp_noise = torch.normal(torch.zeros_like(similarity_matrix), torch.full_like(similarity_matrix, std)).to(self.args.device)
        # similarity_matrix+=dp_noise
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
    
    def compute_loss(self, features, features_prime, projection):
        logits, labels = self.info_nce_loss(projection)
        l1 = self.criterion(logits, labels)
        l2 = self.correlationLoss(features, features_prime)
        return l1 * 0.50 + l2

    def evaluate(self, val_loader, curr_loss = None, epoch = 0):
        losses = []
        avg_coorelation = []
        with torch.no_grad():
            for images, _ in tqdm(val_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                projection,features = self(images)
                features_prime = ARCH0(images)
                loss = self.compute_loss(features, features_prime, projection)
                losses.append(loss.item())
                csm = torch.cosine_similarity(features_prime, features)
                avg_coorelation.append([csm.mean().item()])
        mean_loss =  np.mean(losses)
        if self.args.wandb: 
            wandb.log({'mean_val_loss' : mean_loss, 'epoch' : epoch})
            table = wandb.Table(data=avg_coorelation, columns=["coorelation"])
            wandb.log({'cosine_distribution_val': wandb.plot.histogram(table, "coorelation", title="Prediction Coorelation Distribution")})
            print('loss saved for epoch : ', epoch)

        # save the best
        if curr_loss:
           if curr_loss >= mean_loss:
               torch.save(self.state_dict(), f'/home2/praguna.manvi/plg_models/model_{self.args.lr}_{self.args.batch_size}.pt')   
               wandb.log({'saving_epoch' : epoch+1})
        return mean_loss


    def train(self, train_loader, val_loader):
        curr_loss = torch.inf
        for epoch in range(self.args.epoch):
            for images,_ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                projection,features = self(images)
                features_prime = ARCH0(images)
                loss = self.compute_loss(features, features_prime, projection)
                # print(loss.item())
                if self.args.wandb: wandb.log({'loss' : loss.item(), 'epoch' : epoch+1})
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
            curr_loss = self.evaluate(val_loader, curr_loss, epoch+1)


    def forward(self, x):
        x = self.backend_layer(x)['back_out0'].view(-1, 1792)
        y = self.summaryVec(x)
        x = self.nonLinearOut(y)
        return x,y 

    def private_compute_loss(self, model, features, features_prime, projection):
        logits, labels = self.info_nce_loss(projection)
        l1 = model.criterion(logits, labels)
        l2 = model.correlationLoss(features, features_prime)
        return l1 + l2 
    
    def private_train(self, model, optimizer, train_loader, privacy_engine):
        losses = []
        for epoch in range(self.args.epoch):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                projection,features = model(images)
                features_prime = ARCH0(images)
                loss = self.private_compute_loss(model, features, features_prime, projection)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epsilon = privacy_engine.accountant.get_epsilon(delta=self.args.delta)
                
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"(ε = {epsilon:.4f}, δ = {self.args.delta})"
            )

# if __name__ == "__main__":
#     # remove this 
#     device  = 'cuda' if torch.cuda.is_available() else 'cpu'
#     import argparse
#     parser = argparse.ArgumentParser(description='Nvgnet')
#     parser.add_argument('--temperature', default=0.07, type=float,
#                     help='softmax temperature (default: 0.07)')
#     parser.add_argument('--device', default='cuda', type=str,
#                     help='device type')
#     parser.add_argument('--lr', default=0.0003, type=float,
#                     help='learning rate')
#     parser.add_argument('--weight_decay', default=0.0001, type=float,
#                     help='weight_decay')
#     parser.add_argument('--epoch', default=1, type=int,
#                     help='epoch')
#     parser.add_argument('--batch_size', default=2, type=int,
#                     help='batch_size')
#     parser.add_argument('--delta', default=1e-5, type=float,
#                     help='delta')
#     args = parser.parse_args()
#     import time
#     nvgNetFace = NvgnetFace(args=args).to(device)
#     in0 = torch.randn([64, 3, 160, 160]).to(device)
#     s = time.time()
#     assert nvgNetFace(in0)[0].shape == torch.Size([64, 128])
#     e = time.time()
#     print(e - s)
#     # private validation and initialization of required params
#     from opacus.validators import ModuleValidator
#     import sys
#     sys.setrecursionlimit(100000)
#     nvgNetFace = ModuleValidator.fix(nvgNetFace)
#     # nvgNetFace.backend_layer = ModuleValidator.fix(nvgNetFace.backend_layer)
#     errors = ModuleValidator.validate(nvgNetFace, strict=False)
#     print(len(errors))
#     nvgNetFace = nvgNetFace.to(device)
#     s = time.time()
#     assert nvgNetFace(in0)[0].shape == torch.Size([64, 128])
#     e = time.time()
#     print(nvgNetFace.summaryVec, nvgNetFace.backend_layer.conv2d_1a)
#     print(e - s)
#     nvgNetFace.init_optimizer()
#     # testing the training loop
#     dummyLoader = [[torch.randn([2, 3, 168, 168]), torch.randn([2, 3, 168, 168])]]
#     nvgNetFace.train(dummyLoader)
#     # train privately
#     from torch.utils.data import TensorDataset, DataLoader, Dataset
#     import numpy as np
#     dataset = TensorDataset(torch.randn([1, 3, 168, 168]), torch.randn([1, 3, 168, 168]), torch.randn([1, 3, 168, 168]), torch.randn([1, 3, 168, 168])) # create your datset
#     dataloader = DataLoader(dataset, batch_size=4) # create your dataloader
#     from opacus import PrivacyEngine
#     privacy_engine = PrivacyEngine()
#     nvgNetFacePrivate, optimizer, train_loader = privacy_engine.make_private(
#         module=nvgNetFace,
#         optimizer=nvgNetFace.optimizer,
#         data_loader=dataloader,
#         noise_multiplier=1.1,
#         max_grad_norm=1.0,
#     )
#     # print(nvgNetFacePrivate)
#     nvgNetFace.private_train(nvgNetFacePrivate, optimizer, train_loader)