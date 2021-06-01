from torchvision import datasets
from torchvision.transforms import ToTensor
from dist_dataloader import dataloader
import ray
ray.init(address="auto")
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

mnist_loader = dataloader.DistDataLoader(training_data, num_workers=4, shuffle=True)

for item in mnist_loader:
    print(item)

