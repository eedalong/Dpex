## Dpex——用户无感知分布式数据预处理组件
### 一、前言

长久以来，深度学习模型训练的分布式更多关注的是分布式GPU计算，对

#### 1.1 数据预处理横向扩展
#### 1.2 GPU资源和CPU资源使用解耦
### 二、架构介绍（介绍Pytorch DataLoader本身的架构以及DistDataLoader的架构）

### 三、使用示例（展示在单卡训练，多卡训练时的使用示例）
#### 3.1 单卡训练
    from torch.utils.data import Dataset
    from Dpex import dataloader
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
    train_loader = dataloader.DpexDataLoader(bug_dataset, distribute_mode=True, num_workers=10, batch_size=100, shuffle=True)
    for epoch in range(5):
        for item in train_loader:
            print(item)
#### 3.2 基于DataParallel的多卡训练
如果你想在

    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from torch.utils.data import Dataset
    from Dpex import dataloader
    
    
    input_size = 5
    output_size = 2
    batch_size = 30
    data_size = 30
    
    class RandomDataset(Dataset):
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size)
    
        def __getitem__(self, index):
            return self.data[index]
    
        def __len__(self):
            return self.len
    
    rand_loader = dataloader.DpexDataLoader(dataset=RandomDataset(input_size, data_size),
                                            distribute_mode=True, batch_size=batch_size, shuffle=True, num_workers=10)
    
    class Model(nn.Module):
        # Our model
    
        def __init__(self, input_size, output_size):
            super(Model, self).__init__()
            self.fc = nn.Linear(input_size, output_size)
    
        def forward(self, input):
            output = self.fc(input)
            print("  In Model: input size", input.size(),
                  "output size", output.size())
            return output
    model = Model(input_size, output_size)
    
    if torch.cuda.is_available():
        model.cuda()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)
    
    for data in rand_loader:
        if torch.cuda.is_available():
            input_var = Variable(data.cuda())
        else:
            input_var = Variable(data)
        output = model(input_var)
        print("Outside: input size", input_var.size(), "output_size", output.size())

#### 3.3 基于DDP的多卡训练
### 四、Benchmark（展示使用DistDataLoader后模型精度没有影响，同时对部分数据预处理比较重的模型训练有明显加速）
#### 4.1 模型精度Benchmark

#### 4.2 训练速度Benchmark
### 五、环境依赖:
#### 5.1 Ray集群搭建



