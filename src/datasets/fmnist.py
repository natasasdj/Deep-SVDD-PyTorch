from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import FashionMNIST as FMNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms


class FMNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class

        min_max =  [(-2.681241512298584, 24.85430335998535), 
                    (-2.5778563022613525, 11.169791221618652), 
                    (-2.808171272277832, 19.133548736572266), 
                    (-1.9533655643463135, 18.656728744506836), 
                    (-2.610386610031128, 19.166683197021484), 
                    (-1.2358512878417969, 28.463111877441406), 
                    (-3.2516062259674072, 24.19683265686035), 
                    (-1.0814448595046997, 16.878820419311523), 
                    (-3.656099319458008, 11.350275993347168), 
                    (-1.3859291076660156, 11.426652908325195)]

        # FMNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyFMNIST(root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyFMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)


class MyFMNIST(FMNIST):
    """Torchvision FMNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyFMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the FMNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
