import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100

class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 **kwargs,
                 ):

        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes
        
        self.default_class_order = [70, 89, 11, 13, 63, 53, 86, 57, 41, 43, 14, 98, 52, 73, 95, 96, 33, 16, 39, 74, 25, 88, 35, 28, 79, 82, 72, 4, 30, 17, 59, 97, 36, 38, 29, 55, 83, 7, 22, 48, 19, 47, 2, 44, 67, 71, 34, 84, 6, 46, 61, 8, 80, 10, 49, 15, 68, 9, 99, 40, 27, 45, 51, 37, 21, 64, 92, 24, 60, 31, 5, 91, 93, 90, 65, 66, 77, 20, 58, 62, 23, 76, 75, 42, 0, 26, 87, 50, 3, 56, 81, 1, 94, 69, 18, 78, 54, 12, 85, 32]
