# 模型迁移：PyTorch 到 MindSpore

本项目展示了如何将 PyTorch 实现的神经网络模型迁移到 MindSpore 框架。通过一个简单的多层感知机（MLP）模型，演示了两个深度学习框架之间的关键区别和迁移流程。

## 模型架构

本项目实现了一个简单的多层感知机 (MLP) 模型，具有以下架构：

- **输入层**：784个神经元（适用于28x28的图像，如MNIST数据集）
- **第一隐藏层**：128个神经元，使用ReLU激活函数
- **第二隐藏层**：64个神经元，使用ReLU激活函数
- **输出层**：10个神经元，使用LogSoftmax激活函数（适用于10类分类问题）

```
输入 (784) → 隐藏层1 (128) → 隐藏层2 (64) → 输出 (10)
```

## 应用场景

该模型适用于多种基础分类任务，特别是：

1. **手写数字识别**：可以在MNIST数据集上训练该模型，识别0-9的手写数字。
2. **简单图像分类**：可以应用于低分辨率、灰度图像的基础分类任务。
3. **教育与学习**：作为深度学习入门的示例模型，学习神经网络的基本概念。
4. **框架迁移研究**：比较不同深度学习框架的实现差异和性能特点。

虽然这是一个简单的模型，但它反映了深度学习的核心概念，并为更复杂模型的实现和迁移提供了基础。

## 框架迁移

### 主要差异

将模型从PyTorch迁移到MindSpore时，存在以下几个关键差异：

1. **模型定义**：
   - PyTorch: 使用`nn.Module`作为基类，`forward`方法定义前向传播
   - MindSpore: 使用`nn.Cell`作为基类，`construct`方法定义前向传播

2. **层命名**：
   - PyTorch: 使用`nn.Linear`作为全连接层
   - MindSpore: 使用`nn.Dense`作为全连接层

3. **执行模式**：
   - PyTorch: 默认使用动态图模式
   - MindSpore: 需要明确设置执行模式，如`context.set_context(mode=context.GRAPH_MODE)`

4. **数据处理**：
   - PyTorch: 使用`DataLoader`加载数据
   - MindSpore: 使用专有的`Dataset`接口

5. **训练流程**：
   - PyTorch: 显式定义训练循环，手动调用`backward()`和`step()`
   - MindSpore: 使用更高级的`Model`API封装训练过程

### 迁移实践经验

1. 首先确保理解原始PyTorch模型的架构和功能
2. 识别PyTorch特有的操作和API，寻找MindSpore中的对应功能
3. 注意两个框架在数据类型、维度处理上的细微差别
4. 使用相同的初始化方法，确保公平比较
5. 迁移后验证模型在相同数据上的性能表现

## 使用说明

### 环境需求

```
# PyTorch实现
torch>=1.8.0

# MindSpore实现
mindspore>=1.5.0
```

### 使用方法

#### PyTorch模型

```python
from pytorch_mlp import SimpleMLP, train_model, evaluate_model

# 创建模型
model = SimpleMLP(input_size=784, output_size=10)

# 训练模型(需要自行准备数据加载器train_loader)
trained_model = train_model(model, train_loader, epochs=5, lr=0.01)

# 评估模型
accuracy = evaluate_model(trained_model, test_loader)
```

#### MindSpore模型

```python
from mindspore_mlp import SimpleMLP, train_model, evaluate_model

# 创建模型
model = SimpleMLP(input_size=784, output_size=10)

# 训练模型(需要自行准备MindSpore格式的train_dataset)
trained_model = train_model(model, train_dataset, epochs=5, lr=0.01)

# 评估模型
accuracy = evaluate_model(trained_model, test_dataset)
```

## 参考资料

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [MindSpore官方文档](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- [深度学习框架迁移指南](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/migrating_from_pytorch_introduction.html)