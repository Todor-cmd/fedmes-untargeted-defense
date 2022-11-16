from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class FashionMNISTDataset(Dataset):

    def __init__(self, args):
        super(FashionMNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST train data")

        train_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST train data")

        return train_data

    def load_ba_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST ba data")

        train_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_dataset.targets[train_dataset.targets > 0] = 0
        for img in train_dataset.data:
            img[0][0] = 255
            img[27][0] = 255
        ba_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        ba_data = self.get_tuple_from_data_loader(ba_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST ba data")

        return ba_data

    def load_dba_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST dba data")

        train_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_dataset.targets[train_dataset.targets > 0] = 0
        for img in train_dataset.data[:int(0.5 * len(train_dataset))]:
            img[0][0] = 255
        for img in train_dataset.data[int(0.5 * len(train_dataset)):]:
            img[27][0] = 255
        dba_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        dba_data = self.get_tuple_from_data_loader(dba_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST dba data")

        return dba_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST test data")

        test_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")

        return test_data

    def load_backdoor_test_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST test data")

        test_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset.targets[test_dataset.targets > 0] = 0
        for img in test_dataset.data:
            img[0][0] = 255
            img[27][0] = 255
        backdoor_test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")

        return test_data

    # def load_free_dataset(self):
    #     self.get_args().get_logger().debug("Loading Fashion MNIST free data")
    #
    #     train_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True,
    #                                           transform=transforms.Compose([transforms.ToTensor()]))
    #     benign_dataset, malicious_dataset = torch.utils.data.random_split(train_dataset, [59400, 600])
    #
    #     train_loader = DataLoader(malicious_dataset, batch_size=len(train_dataset))
    #
    #     train_data = self.get_tuple_from_data_loader(train_loader)
    #
    #     self.get_args().get_logger().debug("Finished loading Fashion MNIST free data")
    #
    #     return train_data
