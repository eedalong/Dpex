## DistDataLoader——用户无感知分布式数据预处理组件
### 一、前言（介绍使用场景和动机）
#### 1.1 数据预处理横向扩展
#### 1.2 GPU资源和CPU资源使用解耦
### 二、架构介绍（介绍Pytorch DataLoader本身的架构以及DistDataLoader的架构）

### 三、使用示例（展示在单卡训练，多卡训练时的使用示例）
#### 3.1 单卡训练
#### 3.2 基于DataParallel的多卡训练
#### 3.3 基于DDP的多卡训练
### 四、Benchmark（展示使用DistDataLoader后模型精度没有影响，同时对部分数据预处理比较重的模型训练有明显加速）
#### 4.1 模型精度Benchmark
#### 4.2 训练速度Benchmark
### 五、环境依赖:
#### 5.1 Ray集群搭建



