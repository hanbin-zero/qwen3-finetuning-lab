# 阶段一：PyTorch 基础学习与练习

目标：能自己解释并改写“数据 → 模型 → loss → 梯度 → 参数更新 → 验证 → 保存与恢复”的全过程。完成后再进入极小语言模型阶段。

本阶段不需要 Qwen 权重，不下载数据集，不部署服务。示例默认 CPU，后续可在自己准备好的独立 GPU 环境运行。

## 1. 使用方式与准备

配套完整代码：[exercises/stage1_pytorch.py](../exercises/stage1_pytorch.py)。下文代码用于逐步理解；完整脚本是可运行入口，不必把片段拼起来。

示例只依赖 Python 标准库和 PyTorch。运行前在你选择的、已经装好 PyTorch 的环境检查：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

如果不能导入 torch，先记录解释器路径和报错，暂停运行；安装环境另行安排，不在现有服务环境里直接安装或升级。CPU 练习不要求 CUDA。

练习统一采用一个简单任务：输入两个数字 `x1, x2`，若 `x1 + 0.5*x2 > 0` 则标签为 1，否则为 0。人工规则只负责生成答案，神经网络通过样本学习它。这样可以把注意力放在训练机制上。

先自己预测每段代码的结果，再运行核对；每一步完成“能跑、能解释、能改”后进入下一步。

| 步骤 | 主题 | 最少成果 |
| --- | --- | --- |
| 1 | 张量、形状、类型、设备 | 手算一个矩阵乘法与广播结果 |
| 2 | 自动求导与梯度清零 | 手算梯度，观察累积，完成一次更新 |
| 3 | Dataset 与 DataLoader | 正确打印一个 batch 的输入与标签 |
| 4 | 模型与交叉熵 | 解释 logits、标签及 loss 的对应关系 |
| 5 | 完整训练与验证 | 独立记录训练/验证指标 |
| 6 | 小样本过拟合与对照 | 32 条样本达到至少 95% 训练准确率 |
| 7 | 保存、加载、断点恢复 | 预测一致，恢复训练的步数继续增长 |
| 8 | 汇总验收 | 可复现的代码、指标曲线与练习记录 |

## 2. 第一步：张量、形状、类型与设备

**先学**：张量是带形状、数据类型和设备的多维数组；理解 `shape`、`dtype`、`device`、索引、广播、`@` 和逐元素 `*`。

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])  # [3, 2]
w = torch.ones(2, 2)                            # [2, 2]
b = torch.tensor([10., 20.])                    # [2]
result = x @ w + b                             # [3, 2]
print(x.shape, x.dtype, x.device)
print(result)
assert result.shape == (3, 2)
assert torch.equal(result[0], torch.tensor([13., 23.]))
```

**练习**：手算第一行；把输入改成 `[5, 2]`，判断结果形状；尝试 `x * w`，解释这里为什么形状不兼容。再把标签构造成 `torch.tensor([0, 1, 0], dtype=torch.long)`，区分连续特征和类别索引。

**验收**：能解释 `x @ w` 的内维必须一致、为什么 `[2]` 偏置能加到每行、分类标签为什么用整型。能说出“形状错误”和“CPU/GPU 设备不一致”是两类问题。

## 3. 第二步：自动求导与一次参数更新

**先学**：`requires_grad`、计算图、`backward()`、叶子张量的 `.grad`，以及梯度下降。优化器更新的是参数，反向传播计算的是梯度。

```python
import torch

w = torch.tensor(2., requires_grad=True)
loss = (w - 1).square()       # loss=1，导数为 2*(w-1)=2
loss.backward()
print(w.grad)                # 2
(w - 1).square().backward()   # 新建计算图，梯度仍累积到同一个 w
print(w.grad)                # 4
w.grad = None
(w - 1).square().backward()   # 清零后重新得到 2
with torch.no_grad():
    w -= 0.1 * w.grad        # 2 - 0.1*2 = 1.8
assert torch.isclose(w, torch.tensor(1.8))
```

**练习**：把学习率改成 0.5，先手算结果再执行；解释为什么更新参数时关闭梯度。不要对同一个已释放的 `loss` 图反复调用 backward 来演示累积，示例每次都重建图。

**验收**：能给出 `loss=1 → grad=2 → 参数1.8` 的手算过程；能解释“不清梯度”与“有意做梯度累积”的差别。训练代码中的 `optimizer.zero_grad(set_to_none=True)` 对应这里的清零步骤。

## 4. 第三步：Dataset 与 DataLoader

**先学**：单样本、batch、epoch 和 step。Dataset 负责“第 i 条数据是什么”，DataLoader 负责取样、打乱和组 batch。

在完整脚本中阅读 `PointsDataset`，其核心是：

```python
generator = torch.Generator().manual_seed(100)
x = torch.randn(512, 2, generator=generator)
y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
```

随后使用自定义 Dataset 中的 `__len__`、`__getitem__`，通过 DataLoader 读取：

```python
from torch.utils.data import DataLoader
from stage1_pytorch import PointsDataset

dataset = PointsDataset(512, seed=100)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
xb, yb = next(iter(loader))
assert xb.shape == (32, 2)
assert yb.shape == (32,)
assert yb.dtype == torch.int64
print(len(dataset), len(loader))  # 512 条数据，16 个 batch
```

该片段在 `exercises` 目录执行，才能直接导入同目录脚本；完整脚本本身不依赖此导入步骤。`num_workers=0` 便于先排除多进程问题。

**练习**：batch size 改成 64，判断每轮 step 数；数据条数改成 513，判断最后一个 batch 的大小，检查 `drop_last` 默认行为。

**验收**：能说明单条输入 `[2]`、batch 输入 `[32, 2]`、标签 `[32]` 的关系；知道训练集可打乱，验证集不能拿来反向传播。示例用不同种子独立生成 512 条训练数据和 256 条验证数据，没有跨集合复用样本。

## 5. 第四步：模型、logits 与 loss

**先学**：`nn.Module`、`nn.Linear`、ReLU、参数、logits 与交叉熵。此处两个输出分别是类别 0 和类别 1 的分数。

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
x = torch.randn(32, 2)
y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
logits = model(x)
loss = nn.functional.cross_entropy(logits, y)
predicted = logits.argmax(dim=1)
assert logits.shape == (32, 2)
assert loss.ndim == 0
print(loss.item(), (predicted == y).float().mean().item())
print(sum(p.numel() for p in model.parameters()))  # 82
```

**练习**：手算参数量 `(2*16+16)+(16*2+2)=82`；隐藏层宽度改成 8，重新算。理解交叉熵直接接收 logits，这里不在模型末尾加 softmax。

**验收**：能解释为什么二分类可以输出两个 logits、为什么标签是 `[32]` 而不是 `[32, 2]`；能区分 loss 和准确率，不把 `.item()` 得到的普通数字用于反向传播。

## 6. 第五步：完整训练和验证

**先学**：训练模式、评测模式、禁用梯度、按样本数加权汇总指标。`eval()` 改变 Dropout/BatchNorm 等层的行为，本身不会关闭自动求导。

核心训练顺序，见完整脚本的 `train_epoch`：

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(xb)
loss = nn.functional.cross_entropy(logits, yb)
loss.backward()
optimizer.step()
```

验证要用 `model.eval()` 和 `torch.no_grad()`，不调用 backward 或 step。完整脚本按 `loss * batch样本数` 累计，再除以总样本数，避免最后一个小 batch 被赋予同等权重。

在示例目录用已具备 PyTorch 的环境运行：

```bash
python stage1_pytorch.py --device cpu --out run_cpu
```

示例执行张量/梯度自检、32 条数据过拟合，再训练 20 个 epoch，每轮打印 `train_loss/train_acc/val_loss/val_acc`。输出 `metrics.csv` 和 `last.pt` 到指定本地目录，重复使用同一目录会覆盖已有结果，因此不同实验使用不同目录。

**练习**：在代码副本中分别只改学习率或 batch size，记录变化。学习率变大不保证更好；逐轮 loss 也不保证严格单调下降。

**验收**：固定默认配置完成 20 轮；无 NaN；验证准确率至少 90%，验证 loss 低于初始值。这个门槛只适用于本合成任务，不外推到金融文本或 Qwen。

## 7. 第六步：小样本过拟合与受控对照

**先学**：固定小批数据过拟合用于确认模型、标签、梯度链路能工作；它不检验泛化。

完整脚本的 `overfit_demo` 固定 32 条样本，使用不含 Dropout 的模型训练 200 次，检查准确率至少 95%，并确认参数确实改变。

**练习与观察**

| 对照 | 做法 | 应观察什么 |
| --- | --- | --- |
| 梯度链路 | 在副本暂时去掉 `optimizer.step()` | 参数不更新；准确率通常无法靠训练提高，脚本自检应失败 |
| 模式差别 | 单独构造 `nn.Dropout(0.5)`，对全 1 张量分别用 train/eval 前向 | train 随机置零并缩放，eval 保持输入；不用“随机两次一定不同”做断言 |
| 记忆与泛化 | 用较大隐藏层拟合 32 条随机标签，验证集使用独立随机标签 | 可能记住训练数据，但验证集不能稳定优于随机水平；记录实际结果，不保证一次达到某分数 |

**验收**：先用原脚本通过 95% 小样本门槛，再说明每个对照改变了哪一个变量。能列出过拟合失败时的检查顺序：标签与形状 → 参数是否进优化器 → 梯度是否存在 → step 是否执行 → 学习率与模式。

## 8. 第七步：保存、加载与断点恢复

**先学**：`state_dict`、模型结构、优化器状态、训练位置与随机状态。只保存权重适合推理；继续训练还需要优化器等状态。

完整脚本每个 epoch 结束保存 `last.pt`：模型、Adam 状态、epoch/global_step、配置、指标、Python/PyTorch 随机状态、DataLoader shuffle 生成器状态，以及 CUDA 状态（使用 CUDA 时）。示例只支持 **epoch 边界恢复**，不覆盖任意 batch 中断。

用三个独立进程比较连续训练与恢复训练，均在本地 CPU 执行：

```bash
python stage1_pytorch.py --device cpu --epochs 20 --out continuous
python stage1_pytorch.py --device cpu --epochs 20 --stop-after 10 --out interrupted
python stage1_pytorch.py --device cpu --epochs 20 --resume interrupted/last.pt --out resumed
```

第三次应打印 `resumed after epoch 10`，从第 11 轮继续，最终 `global_step=320`（每轮 16 个 batch，共 20 轮）。恢复后的 CSV 保留前 10 轮记录。

检查两个最终检查点，以下代码在上述工作目录执行：

```python
import torch

a = torch.load('continuous/last.pt', map_location='cpu', weights_only=True)
b = torch.load('resumed/last.pt', map_location='cpu', weights_only=True)
assert a['epoch'] == b['epoch'] == 20
assert a['global_step'] == b['global_step'] == 320
for name in a['model']:
    torch.testing.assert_close(a['model'][name], b['model'][name])
print('PASS: continuous and resumed parameters match')
```

加载用 `weights_only=True`，示例检查点只包含张量和基本类型；使用自己生成的检查点，不以修改加载参数绕过陌生文件的加载错误。

**验收**：原脚本中“新模型实例加载后固定输入 logits 一致”的断言通过；独立进程恢复完成；模型参数对照通过。本示例无 Dropout、无多 worker，CPU 测试便于验证；GPU、不同版本或更复杂训练不承诺逐位一致。

## 9. 第八步：最终验收与记录

保留代码版本、Python/PyTorch 版本、运行命令、设备、学习率、batch size、种子、CSV 和检查点。用 CSV 画训练/验证 loss 曲线，并列出训练/验证准确率；可用已有绘图工具，不为本练习修改服务器环境。

如本地已有 matplotlib，可以在结果目录运行以下代码生成图片；它是可选可视化，不是主脚本依赖：

```python
import csv
import matplotlib.pyplot as plt

with open('metrics.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
epochs = [int(r['epoch']) for r in rows]
for key in ['train_loss', 'val_loss']:
    plt.plot(epochs, [float(r[key]) for r in rows], label=key)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss.png', dpi=150)
```

最终自查：

- [ ] 不看代码说清一次训练 step 的顺序，以及每一步的作用。
- [ ] 手算张量乘法、标量梯度和模型参数量。
- [ ] Dataset/batch/logits/标签形状正确，张量和梯度自检通过。
- [ ] 32 条样本训练准确率 ≥95%，知道它不能证明泛化。
- [ ] 默认配置验证准确率 ≥90%，验证 loss 相对初始下降。
- [ ] 训练/验证指标分开，有曲线、配置和至少一个对照实验记录。
- [ ] 加载预测一致，独立进程恢复训练和参数对照通过。
- [ ] 能解释训练变好、验证变差时应先检查什么。

以上全部完成再进入阶段二。若失败，保留报错和配置，先定位失败环节，不用更大的模型或显卡掩盖问题。

### 配套脚本的本地验证记录

2026-10-29，在本地 Python 3.11 / PyTorch `2.7.0+cpu` 验证默认配置：32 条样本过拟合准确率 100%；20 轮结束时验证准确率约 98.0%，验证 loss 约 0.0526；重新加载预测一致。另用独立进程从第 10 轮恢复到第 20 轮，最终模型参数和完整指标历史与连续训练完全一致。

以上只证明该示例在这一套本地 CPU 环境通过，不代表服务器 GPU 环境、Qwen 或金融任务已验证。仓库不包含模型检查点；可按文中命令重新生成结果，公开验证摘要见 [CPU 验证报告](../reports/stage1-cpu.md)。

## 10. 后续 GPU 与模型入口

阶段一 CPU 足够。后续先核对独立环境和空闲 GPU，再选择可见设备。下面仅以第 0 张卡为示例，在 Bash 中运行；请按自己的硬件修改编号：

```bash
CUDA_VISIBLE_DEVICES=0 python stage1_pytorch.py --device cuda --out run_gpu
```

进程内 `cuda:0` 对应被暴露的第一张 GPU，不一定是宿主机的第 0 张。示例使用 FP32，不引入混合精度、多卡或模型下载。

阶段三具体使用 **Qwen 官方 `Qwen/Qwen3-1.7B`**（已经过后训练的对话模型，不是 `Qwen3-1.7B-Base`，也不是 GGUF/AWQ 社区量化版本）。型号本身没有 `-Instruct` 后缀，不要自行添加：

- [魔搭：Qwen/Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B)
- [Hugging Face 官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)

## 官方参考

- [PyTorch 张量基础](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [自动求导](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Dataset 与 DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [训练和优化](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [保存和恢复模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
