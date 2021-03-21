from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np

class Lung_Dataset(Dataset):
    
    def __init__(self, groups="train", transforms=None):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.

        groups = ('train', 'val', 'test')
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2: 'infected_non_covid'}
        
        # The dataset consists only of validation images
        self.groups = groups
        self.transforms = transforms
        if groups == 'train':
            # Number of images in each part of the dataset
            self.dataset_numbers = {'train_normal': 1341,\
                                   'train_infected_covid': 1345,\
                                   'train_infected_non_covid': 2530}
            
            # Path to images for different parts of the dataset
            self.dataset_paths = {'train_normal': './dataset/train/normal/',\
                                  'train_infected_covid': './dataset/train/infected/covid',
                                  'train_infected_non_covid': './dataset/train/infected/non-covid'}
        elif groups == 'val':
            # Number of images in each part of the dataset
            self.dataset_numbers = {'val_normal': 8,\
                                   'val_infected_covid': 9,\
                                   'val_infected_non_covid': 8}
            
            # Path to images for different parts of the dataset
            self.dataset_paths = {'val_normal': './dataset/val/normal/',\
                                  'val_infected_covid': './dataset/val/infected/covid',
                                  'val_infected_non_covid': './dataset/val/infected/non-covid'}
        elif groups == 'test':
            # Number of images in each part of the dataset
            self.dataset_numbers = {'test_normal': 234,\
                                   'test_infected_covid': 139,\
                                   'test_infected_non_covid': 242}
            
            # Path to images for different parts of the dataset
            self.dataset_paths = {'test_normal': './dataset/test/normal/',\
                                  'test_infected_covid': './dataset/test/infected/covid',
                                  'test_infected_non_covid': './dataset/test/infected/non-covid'}
        else:
              raise Exception("Groups parameter must be either train, test or val")
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        desc = None
        if self.groups=='train':
            desc = 'training'
        elif self.groups=='val':
            desc = 'validation'
        elif self.groups=='test':
            desc = 'testing'
        msg = f"This is the {desc} dataset of th Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'infected_covid', 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal', 'infected_covid', 'infected_non_covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
#         with open(path_to_file, 'rb') as f:
#            im = np.asarray(Image.open(f))/255
        im = Image.open(path_to_file)
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'infected_covid', 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        
        # Get item special method
        normal_index, infected_covid_index, _ = map(int, list(self.dataset_numbers.values()))
        if index < normal_index:
            class_val = 'normal'
            label = torch.Tensor([1, 0, 0])
        elif index < infected_covid_index + normal_index:
            class_val = 'infected_covid'
            index = index - normal_index
            label = torch.Tensor([0, 1, 0])
        else:
            class_val = 'infected_non_covid'
            index = index - infected_covid_index - normal_index
            label = torch.Tensor([0, 0, 1])
        im = self.open_img(self.groups, class_val, index)

        if self.transforms:
            im = self.transforms(im)
        else:
            im = np.asarray(im)/255
            im = transforms.functional.to_tensor(np.array(im)).float()

        return im, label


class Binary_Lung_Dataset(Dataset):
    
    def __init__(self, groups="train", classify="normal", transforms=None):
        """
        Constructor for Binary Dataset class - simply assembles
        the important parameters in attributes.
        
        groups = ('train', 'val', 'test')
        classify = ('normal', 'infected')
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        self.groups = groups
        self.classify = classify
        self.transforms = transforms
        # Only two classes will be considered here
        if classify == 'normal':
            self.classes = {0: 'normal', 1: 'infected'}
            if groups == 'train':
                # Number of images in each part of the dataset
                self.dataset_numbers = {'train_normal': 1341,\
                                       'train_infected_covid': 1345,\
                                       'train_infected_non_covid': 2530}

                # Path to images for different parts of the dataset
                self.dataset_paths = {'train_normal': './dataset/train/normal/',\
                                      'train_infected_covid': './dataset/train/infected/covid',
                                      'train_infected_non_covid': './dataset/train/infected/non-covid'}
            elif groups == 'val':
                # Number of images in each part of the dataset
                self.dataset_numbers = {'val_normal': 8,\
                                       'val_infected_covid': 9,\
                                       'val_infected_non_covid': 8}

                # Path to images for different parts of the dataset
                self.dataset_paths = {'val_normal': './dataset/val/normal/',\
                                      'val_infected_covid': './dataset/val/infected/covid',
                                      'val_infected_non_covid': './dataset/val/infected/non-covid'}
            elif groups == 'test':
                # Number of images in each part of the dataset
                self.dataset_numbers = {'test_normal': 234,\
                                       'test_infected_covid': 139,\
                                       'test_infected_non_covid': 242}

                # Path to images for different parts of the dataset
                self.dataset_paths = {'test_normal': './dataset/test/normal/',\
                                      'test_infected_covid': './dataset/test/infected/covid',
                                      'test_infected_non_covid': './dataset/test/infected/non-covid'}
            else:
                  raise Exception("Groups parameter must be either train, test or val")
            
        elif classify == 'infected':
            self.classes = {0: 'infected_covid', 1: 'infected_non_covid'}
            if groups == 'train':
                # Number of images in each part of the dataset
                self.dataset_numbers = {'train_infected_covid': 1345,\
                                       'train_infected_non_covid': 2530}

                # Path to images for different parts of the dataset
                self.dataset_paths = {'train_infected_covid': './dataset/train/infected/covid',
                                      'train_infected_non_covid': './dataset/train/infected/non-covid'}
            elif groups == 'val':
                # Number of images in each part of the dataset
                self.dataset_numbers = {'val_infected_covid': 9,\
                                       'val_infected_non_covid': 8}

                # Path to images for different parts of the dataset
                self.dataset_paths = {'val_infected_covid': './dataset/val/infected/covid',
                                      'val_infected_non_covid': './dataset/val/infected/non-covid'}
            elif groups == 'test':
                # Number of images in each part of the dataset
                self.dataset_numbers = {'test_infected_covid': 139,\
                                       'test_infected_non_covid': 242}

                # Path to images for different parts of the dataset
                self.dataset_paths = {'test_infected_covid': './dataset/test/infected/covid',
                                      'test_infected_non_covid': './dataset/test/infected/non-covid'}
            else:
                  raise Exception("Groups parameter must be either train, test or val")
        else:
            raise Exception("classify parameter must be either normal or infected")
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        desc = None
        if self.groups=='train':
            desc = 'training'
        elif self.groups=='val':
            desc = 'validation'
        elif self.groups=='test':
            desc = 'testing'
        msg = f"This is the {desc} dataset of th Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'infected_covid', 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
#         with open(path_to_file, 'rb') as f:
#             im = np.asarray(Image.open(f))/255
#             im = Image.open(f)
        im = Image.open(path_to_file)
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'infected_covid', 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        
        # Get item special method
        if self.classify == 'normal':
            normal_index, infected_covid_index, _ = map(int, list(self.dataset_numbers.values()))
            if index < normal_index:
                class_val = 'normal'
                label = torch.Tensor([1, 0])
            elif index < infected_covid_index + normal_index:
                class_val = 'infected_covid'
                index = index - normal_index
                label = torch.Tensor([0, 1])
            else:
                class_val = 'infected_non_covid'
                index = index - infected_covid_index - normal_index
                label = torch.Tensor([0, 1])
        else:
            first_index = int(list(self.dataset_numbers.values())[0])
            if index < first_index:
                class_val = 'infected_covid'
                label = torch.Tensor([1, 0])
            else:
                class_val = 'infected_non_covid'
                index = index - first_index
                label = torch.Tensor([0, 1])
        im = self.open_img(self.groups, class_val, index)

        if self.transforms:
            im = self.transforms(im)
        else:
            im = np.asarray(im)/255
            im = transforms.functional.to_tensor(np.array(im)).float()
            
        return im, label