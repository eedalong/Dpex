from torchvision import datasets
from torchvision.transforms import ToTensor
from dist_dataloader import dataloader
import ray
import torch

# init ray environment
ray.init(address="auto")

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

train_loader = dataloader.DistDataLoader(training_data, distribute_mode=True, num_workers=10, batch_size=100, shuffle=True)
test_loader = dataloader.DistDataLoader(training_data, distribute_mode=True, num_workers=1, batch_size=100, shuffle=False)

num_epochs = 5
for epoch in range(num_epochs):
    print(f"epoch is {epoch}")
    index = 0
    for image, label in train_loader:
        if index % 1000 == 0:
            print(f"check item index = {index}")
        index += 1



