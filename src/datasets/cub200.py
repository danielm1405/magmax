import os
import torch
import torchvision.datasets as datasets


class CUB200:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 **kwargs,
                ):
        # Data loading code
        traindir = os.path.join(location, 'CUB_200_2011_splitted/images_train_test', 'train')
        valdir = os.path.join(location, 'CUB_200_2011_splitted/images_train_test', 'val')

        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # [4:] to trim XXX. prefix
        self.classnames = [c.replace('_', ' ')[4:] for c in list(self.train_dataset.class_to_idx.keys())]
    
        self.default_class_order = [186, 157, 67, 59, 117, 1, 122, 14, 131, 40, 66, 199, 92, 170, 197, 118, 173, 37, 146, 191, 4, 198, 64, 56, 51, 165, 0, 107, 175, 148, 187, 161, 137, 180, 133, 181, 177, 12, 20, 74, 143, 142, 110, 116, 194, 138, 183, 89, 45, 121, 190, 94, 36, 83, 130, 120, 21, 38, 125, 145, 182, 103, 105, 61, 32, 141, 13, 112, 111, 104, 26, 69, 39, 156, 147, 108, 72, 10, 96, 188, 16, 63, 152, 86, 160, 119, 184, 7, 172, 132, 44, 42, 82, 15, 114, 127, 85, 168, 55, 65, 3, 95, 171, 176, 33, 43, 17, 97, 163, 62, 75, 123, 128, 124, 99, 29, 28, 166, 41, 35, 76, 77, 5, 84, 102, 60, 78, 70, 164, 113, 30, 91, 81, 18, 155, 179, 73, 46, 80, 31, 150, 79, 174, 153, 50, 144, 9, 167, 134, 135, 101, 49, 106, 154, 23, 100, 48, 25, 71, 139, 162, 158, 159, 87, 90, 11, 24, 57, 19, 52, 129, 140, 178, 53, 169, 195, 126, 47, 151, 136, 22, 58, 34, 8, 185, 54, 109, 193, 68, 192, 149, 115, 98, 88, 93, 6, 196, 189, 2, 27]


class CUB200CustomTemplates(CUB200):
    pass
