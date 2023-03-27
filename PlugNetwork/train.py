import torch
from view_generator import *
from network import *
from opacus.validators import ModuleValidator
import sys
from opacus import PrivacyEngine

# torch.manual_seed(0)


if __name__ == '__main__':
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
    parser.add_argument('--batch_size', default=32, type=int,
                    help='batch_size')
    parser.add_argument('--delta', default=1e-5, type=float,
                    help='delta')
    args = parser.parse_args()
    ffhq_path = '../dumps/thumbnails128x128/' # replace this to an args param
    # dataloader code
    dataset = ContrastiveLearningDataset().get_dataset(ffhq_path, 2)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # model code
    nvgNetFace = NvgnetFace(args=args).to(device)
    sys.setrecursionlimit(100000)
    nvgNetFace = ModuleValidator.fix(nvgNetFace)
    errors = ModuleValidator.validate(nvgNetFace, strict=False)
    print('num validation errors: ', len(errors))
    nvgNetFace = nvgNetFace.to(device)
    nvgNetFace.init_optimizer()
    privacy_engine = PrivacyEngine()
    nvgNetFacePrivate, optimizer, train_loader = privacy_engine.make_private(
        module=nvgNetFace,
        optimizer=nvgNetFace.optimizer,
        data_loader=data_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        poisson_sampling=False
    )
    # print(nvgNetFacePrivate)
    # for images, _ in train_loader:
    #     images = torch.cat(images, dim=0)
    #     print(images.shape)
    #     exit(0)

    nvgNetFace.private_train(nvgNetFacePrivate, optimizer, data_loader, privacy_engine)