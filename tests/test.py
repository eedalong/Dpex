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

for epoch in range(3):
    for index, (image, label) in enumerate(train_loader):
        if index % 100 == 0:
            print(f"epoch_{epoch}:\titerations_{index}")
    time.sleep(20)



