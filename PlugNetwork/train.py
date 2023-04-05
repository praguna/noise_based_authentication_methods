import torch
from view_generator import *
from network import *
from opacus.validators import ModuleValidator
import sys
from opacus import PrivacyEngine
import wandb
wandb.login()

torch.manual_seed(0)


if __name__ == '__main__':
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
    ffhq_path = '../dumps/thumbnails128x128/' # replace this to an args param
    # dataloader codes
    dataset = ContrastiveLearningDataset().get_dataset(ffhq_path, 2)
    train_set_size = int(len(dataset) * 0.9)
    val_set_size = int(len(dataset) * 0.1)
    # test_set_size = len(dataset) - train_set_size - val_set_size
    # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size, test_set_size])
    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])
    
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # model code
    # setup logging
    wandb.init(
      # Set the project where this run will be logged
      project="plg_net", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=args.expname, 
      # Track hyperparameters and run metadata
      config={
        "learning_rate": args.lr,
        "architecture": "NVGNET",
        "dataset": "FFHQ",
        "batch_size" : args.batch_size,
        "epochs": args.epoch,
      })
    # logs inside the next function
    nvgNetFace = NvgnetFace(args=args).to(device)
    # load if required
    # file = f'/home2/praguna.manvi/plg_models/model_{args.lr}_{args.batch_size}.pt'
    # nvgNetFace.load_state_dict(torch.load(file))
    # for n, p in nvgNetFace.named_parameters():
    #     print(n, p.requires_grad)
    ## public training
    nvgNetFace.train(data_loader, val_loader)
    wandb.finish()
    ## private training using DP-ADAM / DP-SGD
    # sys.setrecursionlimit(100000)
    # nvgNetFace = ModuleValidator.fix(nvgNetFace)
    # errors = ModuleValidator.validate(nvgNetFace, strict=False)
    # print('num validation errors: ', len(errors))
    # nvgNetFace = nvgNetFace.to(device)
    # nvgNetFace.init_optimizer()
    # privacy_engine = PrivacyEngine()
    # nvgNetFacePrivate, optimizer, train_loader = privacy_engine.make_private(
    #     module=nvgNetFace,
    #     optimizer=nvgNetFace.optimizer,
    #     data_loader=data_loader,
    #     noise_multiplier=1.1,
    #     max_grad_norm=1.0,
    #     poisson_sampling=False
    # )
    # # print(nvgNetFacePrivate)
    # # for images, _ in train_loader:
    # #     images = torch.cat(images, dim=0)
    # #     print(images.shape)
    # #     exit(0)

    # nvgNetFace.private_train(nvgNetFacePrivate, optimizer, data_loader, privacy_engine)
