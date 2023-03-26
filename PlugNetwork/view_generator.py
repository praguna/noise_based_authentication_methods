from facenet_pytorch import MTCNN
from torchvision import transforms, datasets
import torch
from GaussianBlur import *
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        try:
            return [self.base_transform(x) for i in range(self.n_views)]
        except:
            print('error in image processing')
            return [torch.zeros([3, 160, 160]), torch.zeros([3, 160, 160])] 


class ContrastiveLearningDataset:
    def __init__(self):
        self.mtcnn = MTCNN()

    @staticmethod
    def get_simclr_pipeline_transform(size, tr, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([tr,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              ])
        return data_transforms

    def get_dataset(self, path, n_views):
        return datasets.ImageFolder(root = path, transform = ContrastiveLearningViewGenerator(ContrastiveLearningDataset.get_simclr_pipeline_transform(96, self.mtcnn)))

# # checking speed
# if __name__ == "__main__":
#     import time
#     # preprocess dataset from drive
#     ffhq_path = '../dumps/thumbnails128x128/'
#     dataset = ContrastiveLearningDataset().get_dataset(ffhq_path, 2)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, 
#             num_workers=2, pin_memory=True, drop_last=True)
#     s = time.time()
#     for images, _ in data_loader:
#         images = torch.cat(images, dim=0)
#         print(images.shape)
#         # plt.imsave(f'delete_me.png', images[0].permute(1, 2, 0).numpy())
#         # print(x)
#         break
#     e = time.time()
#     print(e-s)