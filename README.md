## Dpex——用户无感知分布式数据预处理组件
### 一、前言

随着GPU与CPU的算力差距越来越大以及模型训练时的预处理Pipeline变得越来越复杂，CPU部分的数据预处理已经逐渐成为了模型训练的瓶颈所在，这导致单机的GPU配置的提升并不能带来期望的线性加速。预处理性能瓶颈的本质在于每个GPU能够使用的CPU算力受限，为了解决这个问题NVIDIA提出了scale up的方案——使用GPU完成数据预处理，而在这里我们给出scale out的方案——分布式数据预处理库Dpex。

![](blob:https://ukzbv6lwlp.feishu.cn/490e1551-2ba3-403c-8dc0-22d5d2b328cc)
![](blob:https://ukzbv6lwlp.feishu.cn/53871a21-a3b4-4904-a149-ce4d347234f7)
#### 1.1 数据预处理横向扩展
#### 1.2 GPU资源和CPU资源使用解耦
### 二、架构介绍（介绍Pytorch DataLoader本身的架构以及DistDataLoader的架构）

### 三、使用示例（展示在单卡训练，多卡训练时的使用示例）
#### 3.1 单卡训练
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

#### 3.2 基于DataParallel的多卡训练
如果你想在单机上使用DataParallel进行多卡的训练，只需要将Pytorch的DataLoader替换为Dpex中的DataLoader

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
同样，如果你需要在单机上使用DDP进行模型训练，那么核心的代码修改为将Pytorch的DataLoader替换为Dpex的DpexDataLoader
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from Dpex.dataloader import DpexDataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # start command: CUDA_VISIBLE_DEVICES=1,6,7 python -m torch.distributed.launch --nproc_per_node=2 pytorch_ddp.py
    # 1) 初始化
    torch.distributed.init_process_group(backend="nccl")
    
    input_size = 5
    output_size = 2
    batch_size = 1
    data_size = 90000
    
    # 2） 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    class RandomDataset(Dataset):
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size)
    
        def __getitem__(self, index):
            return self.data[index]
    
        def __len__(self):
            return self.len
    
    dataset = RandomDataset(input_size, data_size)
    # 3）使用DistributedSampler
    rand_loader = DpexDataLoader(dataset=dataset, distribute_mode=True, batch_size=batch_size, sampler=DistributedSampler(dataset), num_workers=10)
    
    class Model(nn.Module):
        def __init__(self, input_size, output_size):
            super(Model, self).__init__()
            self.fc = nn.Linear(input_size, output_size)
    
        def forward(self, input):
            output = self.fc(input)
            print("  In Model: input size", input.size(),
                  "output size", output.size())
            return output
    
    model = Model(input_size, output_size)
    
    # 4) 封装之前要把模型移到对应的gpu
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
    
    for data in rand_loader:
        if torch.cuda.is_available():
            input_var = data
        else:
            input_var = data
    
        output = model(input_var)
        print("Outside: input size", input_var.size(), "output_size", output.size())
### 四、Benchmark
在接下来的Benchamark中我们核心展示两个部分的内容:
 - DpexDataLoader对于模型训练精度的影响
 - DpexDataLoader对于模型训练速度的影响 

Dpex只是将单机数据预处理水平扩展到了多机以借助更多的算力来加速数据预处理而不改变数据本身的加载和与处理方式，所以本身对模型的精度不会有负面影响。对于数据预处理较重的情况
#### 4.1 模型精度Benchmark

| **Accuracy**(%) | **Loss** | **GPU Settings** | **DataLoader(If distributed)** | **Epoch** | **Learning rate** | **Batch size** |
| --------------- | -------- | ---------------- | ------------------------------ | --------- | ----------------- | -------------- |
| 90.65           | 0.137    | Single GPU       | True                           | 40        | 0.001             | 100            |
| 91.09           | 0.112    | Single GPU       | False                          | 40        | 0.001             | 100            |
| 90.67           | 0.016    | DataParallel     | True                           | 40        | 0.001             | 100            |
| 90.32           | 0.008    | DataParallel     | False                          | 40        | 0.001             | 100            |
| 88.98           | 0.034    | DDP              | True                           | 40        | 0.001             | 100            |
| 89.84           | 0.030    | DDP              | False                          | 40        | 0.001             | 100            |

#### 4.2 训练速度Benchmark
### 五、环境依赖:
#### 5.1 Ray集群搭建



