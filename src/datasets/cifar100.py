from torch.utils.data import Subset
from PIL import Image
#from torchvision.datasets import CIFAR100
from datasets.coarseCifar100 import CoarseCIFAR100
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms


class CIFAR100_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 20))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max =  [(-14.475378036499023, 12.727412223815918), (-10.135714530944824, 12.160162925720215), 
                    (-10.161286354064941, 14.850489616394043), (-19.340852737426758, 14.995088577270508), 
                    (-9.3675537109375, 12.205577850341797), (-13.702792167663574, 14.258678436279297), 
                    (-14.17085075378418, 10.8519868850708), (-15.715428352355957, 9.566346168518066), 
                    (-7.860762596130371, 8.843354225158691), (-5.972079753875732, 10.153849601745605), 
                    (-9.78101921081543, 8.556866645812988), (-7.734030246734619, 10.450244903564453), 
                    (-10.21424388885498, 9.353658676147461), (-18.856037139892578, 21.160188674926758), 
                    (-8.212591171264648, 10.63957405090332), (-12.311217308044434, 10.493696212768555), 
                    (-7.833277225494385, 10.358002662658691), (-6.648282527923584, 11.645584106445312), 
                    (-7.180142402648926, 11.97570514678955), (-9.8085355758667, 10.46958065032959)]


        # CIFAR-100 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCIFAR100(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCIFAR100(root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)


class MyCIFAR100(CoarseCIFAR100):
    """Torchvision CIFAR100 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR100 class.
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
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
