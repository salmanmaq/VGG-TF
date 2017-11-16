from vgg16 import VGG16
from vgg16 import CIFAR10Loader
import sys

if len(sys.argv) != 2:
    print("Error!")
    sys.exit()

job = sys.argv[1]

# Load the dataset
CIFAR10_data_path = '/media/salman/DATA/General Datasets/cifar-10-batches-py/'
cifarLoader = CIFAR10Loader(CIFAR10_data_path)

# Load the network
network = VGG16(cifarLoader)

if job == 'train':
    network.train()
else:
    network.test()
