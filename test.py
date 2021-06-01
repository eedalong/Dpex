from torchvision import datasets
from torchvision.transforms import ToTensor
from dist_dataloader import dataloader
import ray
ray.init(address="auto")
training_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

mnist_loader = dataloader.DistDataLoader(training_data, num_workers=10, batch_size=128, shuffle=True)

for epoch in range(3):
    print(f"epoch = {epoch}")
    for index, item in enumerate(mnist_loader):
        if index % 1000 == 0:
            print(f"check data index = {index}, check item shape {item[0].shape},  {item[1].shape}")


