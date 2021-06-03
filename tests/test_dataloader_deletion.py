from torchvision import datasets
from torchvision.transforms import ToTensor
from dist_dataloader import dataloader
import time
# init ray environment

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
device = "cpu"

# then we recreate dataloader
train_loader = dataloader.DistDataLoader(training_data, distribute_mode=True, num_workers=10, batch_size=100, shuffle=True)
test_loader = dataloader.DistDataLoader(test_data, distribute_mode=True, num_workers=1, batch_size=100, shuffle=False)

del train_loader
del test_loader