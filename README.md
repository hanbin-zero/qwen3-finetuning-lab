# qwen3-finetuning-lab

以 **smol-course 为实战主线、Qwen3-1.7B 为微调模型**的中文学习项目，逐步掌握 PyTorch、Transformers、Datasets、PEFT 和 TRL。

目标是理解训练机制，完成一次有独立评测、可复现、能解释失败原因的小模型微调。当前实现阶段一，后续阶段按路线逐步补充。

## 学习入口

| 阶段 | 内容 | 当前状态 |
| --- | --- | --- |
| 1 | [PyTorch 基础与训练循环](docs/stage1-pytorch.md) | 教程、练习脚本及 CPU 验证已完成 |
| 2 | 极小语言模型：注意力、因果遮罩与 next-token loss | 已规划 |
| 3 | Qwen3-1.7B：模板、数据处理与 LoRA SFT | 已规划 |
| 4 | 金融文本结构化抽取：基线对照与独立评测 | 已规划 |
| 5 | 适配器推理、资源测量与复现交付 | 已规划 |

完整章节、练习和验收标准见[学习路线](docs/learning-roadmap.md)。推荐顺序是：PyTorch 练习 → 中文原理补充 → Transformers/Datasets 基础 → smol-course Unit 1 → 自定义评测。

## 快速开始：阶段一

阶段一使用合成二维分类数据，默认 CPU，无需下载数据集、模型或调用外部服务。

需要 Python 3.11 和 PyTorch；已验证组合为 Python 3.11 / PyTorch `2.7.0+cpu`。已有兼容环境可直接使用。也可以新建专用环境后安装阶段一依赖：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-stage1.txt
python exercises/stage1_pytorch.py --device cpu --out outputs/stage1
```

脚本检查张量与梯度、小样本过拟合、完整训练/验证和模型重新加载，输出 `metrics.csv` 与 `last.pt`。重复使用同一输出目录会覆盖原结果，独立实验请另选目录。

断点恢复示例（在仓库根目录执行）：

```bash
python exercises/stage1_pytorch.py --device cpu --epochs 20 --stop-after 10 --out outputs/interrupted
python exercises/stage1_pytorch.py --device cpu --epochs 20 --resume outputs/interrupted/last.pt --out outputs/resumed
```

恢复支持 epoch 边界；详细验收和连续训练对照见[阶段一教程](docs/stage1-pytorch.md)。结果证据见[CPU 验证报告](reports/stage1-cpu.md)。

GPU 环境请根据硬件与驱动选择对应 PyTorch 安装包，参考[官方安装入口](https://pytorch.org/get-started/locally/)。本仓库暂未验证 GPU 训练，也未提供阶段三所需五个库的兼容版本锁定文件。

## 模型与实战任务

阶段三使用官方 **`Qwen/Qwen3-1.7B`**，名称没有 `-Instruct` 后缀；它与 `-Base`、社区量化版本不同。

- [魔搭模型页](https://modelscope.cn/models/Qwen/Qwen3-1.7B)
- [官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)

计划以公告/研报片段的主体、事件、日期、金额和证据抽取为任务，比较提示词、少量示例及 LoRA。数据规模、来源划分、指标和采用标准见学习路线。正式样本须来自可公开、许可明确的数据；只完成训练不等于业务效果提升。

## 项目结构

```text
docs/                       分阶段教程与学习路线
exercises/stage1_pytorch.py  阶段一可运行练习
reports/                    去除机器身份信息的验证报告
requirements-stage1.txt     阶段一依赖
```

模型权重、检查点、认证配置、私有数据及实验输出不纳入版本管理。

## 参考与许可

本项目不是 smol-course 的官方版本或完整翻译。教程和练习围绕本项目目标自行组织，章节与 API 参考原项目，引用与许可边界见[参考说明](docs/references.md)。

本仓库原创内容采用 [MIT License](LICENSE)。引用的课程、软件、模型和数据仍遵循各自许可；MIT 不覆盖外部模型权重或数据。
