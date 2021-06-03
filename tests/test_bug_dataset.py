from torch.utils.data import Dataset
from dist_dataloader import dataloader
import time
class BugDataset(Dataset):
    def __init__(self):
        super(BugDataset, self).__init__()
        self.count = 30

    def __getitem__(self, index):
        self.count += 1
        if self.count > 0:
            a = 1 / 0
        time.sleep(1)
        return self.count

    def __len__(self):
        return self.count * 100

bug_dataset = BugDataset()
train_loader = dataloader.DistDataLoader(bug_dataset, distribute_mode=True, num_workers=10, batch_size=100, shuffle=True)
for epoch in range(5):
    for item in train_loader:
        print(item)
