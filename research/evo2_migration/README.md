# Evo2 模型迁移项目

本项目是 ARC Institute 的 Evo2 DNA 语言模型从 PyTorch 框架到 MindSpore 框架的迁移实现接口。

## 关于 Evo2

[Evo2](https://github.com/arcinstitute/evo2) 是 ARC Institute 开发的一个先进的 DNA 语言模型，用于基因组建模和设计。它具有以下主要特点：

1. **长上下文建模**：在单核苷酸分辨率下建模 DNA 序列，上下文长度可达 100 万个碱基对
2. **全领域生命体支持**：在包含 8.8 万亿个标记的 OpenGenome2 数据集上进行训练，覆盖所有生命域
3. **基于 StripedHyena 2 架构**：利用高效的架构来处理长序列
4. **使用 Savanna 预训练**：采用先进的预训练方法

## 模型功能

Evo2 模型支持多种基因组分析和设计任务：

1. **DNA 序列生成**：基于上文提示生成合理的 DNA 序列
2. **序列嵌入提取**：提取 DNA 序列的嵌入表示，用于下游任务
3. **变体效应预测**：预测 DNA 变异的功能效应
4. **基因组特征识别**：识别 DNA 序列中的功能元件和模式

## 框架迁移说明

本项目包含两个接口实现：

1. **PyTorch 接口** (`evo2_torch.py`)：使用 PyTorch 实现的 Evo2 模型接口
2. **MindSpore 接口** (`evo2_mindspore.py`)：使用 MindSpore 框架实现的同一模型接口

迁移过程中主要涉及以下方面的变化：

1. **API 差异**：PyTorch 的 API 调用方式转换为 MindSpore 的对应方法
2. **模型加载和处理**：不同框架下模型权重和配置的处理方式
3. **设备管理**：MindSpore 使用 `ms.set_context()` 设置运行环境，而不是在张量创建时指定 device
4. **数据类型处理**：数据类型转换和处理方式的调整
5. **张量操作转换**：将 PyTorch 的张量操作转换为 MindSpore 的等效操作

## 使用方法

### 环境要求

- PyTorch 接口：需要安装 PyTorch、Vortex 和相关依赖
- MindSpore 接口：需要安装 MindSpore 和相关依赖

### 示例代码

#### DNA序列生成示例

```python
# PyTorch 版本
from evo2_torch import Evo2Model

model = Evo2Model('evo2_7b')
output = model.generate(prompt_seqs=["ACGT"], n_tokens=400, temperature=1.0, top_k=4)
print(output.sequences[0])

# MindSpore 版本
from evo2_mindspore import Evo2Model

model = Evo2Model('evo2_7b')
output = model.generate(prompt_seqs=["ACGT"], n_tokens=400, temperature=1.0, top_k=4)
print(output.sequences[0])
```

#### 嵌入提取示例

```python
# PyTorch 版本
import torch
from evo2_torch import Evo2Model

model = Evo2Model('evo2_7b')
sequence = 'ACGT'
input_ids = torch.tensor(
    model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

layer_name = 'blocks.28.mlp.l3'
outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
print(f'Embeddings shape: {embeddings[layer_name].shape}')

# MindSpore 版本类似实现
```

## 性能对比

两种实现在相同硬件上的性能对比：

| 框架        | 序列生成速度 | 嵌入提取速度 | 内存使用 |
|------------|------------|------------|---------|
| PyTorch    | 基准        | 基准        | 基准     |
| MindSpore  | 略慢        | 略快        | 相似     |

注：实际性能会因硬件环境、序列长度和具体参数设置而有所不同。

## 参考资源

1. [Evo2 官方仓库](https://github.com/arcinstitute/evo2)
2. [Evo2 论文: "Genome modeling and design across all domains of life with Evo 2"](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)
3. [OpenGenome2 数据集](https://huggingface.co/datasets/arcinstitute/opengenome2)
4. [StripedHyena 2 架构](https://github.com/Zymrael/savanna/blob/main/paper.pdf)

## 未来改进

1. 优化模型推理性能，特别是长序列处理
2. 实现更多特定于生物学任务的接口
3. 支持模型量化以减少内存占用
4. 扩展到更多生物信息学应用场景