import torch
import torchvision


TRAIN_DIR = 'data/classification/train/' 
VAL_DIR = 'data/classification/dev/'
TEST_DIR = 'data/classification/test/'

KNOWN_REGEX = 'data/verification/known/*/*'
UNKNOWN_TEST_REGEX = 'data/verification/unknown_test/*'


TRAIN_TRANSFORMS = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.RandomPerspective(distortion_scale=0.2),
                                                   torchvision.transforms.RandomRotation(degrees=10),
                                                   torchvision.transforms.GaussianBlur(kernel_size=3),
                                                   torchvision.transforms.RandomAdjustSharpness(sharpness_factor=1.5),
                                                   torchvision.transforms.RandomAutocontrast(),
                                                   torchvision.transforms.RandomHorizontalFlip(),
                                                   torchvision.transforms.ColorJitter(brightness=0.2,
                                                                                      contrast=0.2,
                                                                                      saturation=0.2,
                                                                                      hue=0.1)])

VAL_TRANSFORMS = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])    

TEST_TRANSFORMS  = VAL_TRANSFORMS

FACE_VERIFICATION_TRANSFORMS = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


CONFIG = {'epochs': 120,
          'batch_size': 256,

          'cosann_start_lr': 0.1,
          'cosann_min_lr': 0.00001,
          
          'sgd_momentum': 0.9,
          'sgd_weight_decay': 0.001,
          
          'cross_entropy_label_smoothing': 0.2}
