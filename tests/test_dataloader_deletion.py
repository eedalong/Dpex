from torchvision import datasets
from torchvision.transforms import ToTensor
from Dpex import dataloader
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
train_loader = dataloader.DpexDataLoader(training_data, distribute_mode=True, num_workers=10, batch_size=100, shuffle=True)
test_loader = dataloader.DpexDataLoader(test_data, distribute_mode=True, num_workers=1, batch_size=100, shuffle=False)

del train_loader
del test_loader