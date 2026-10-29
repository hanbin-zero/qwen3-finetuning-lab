# 阶段一 CPU 验证

验证日期：2026-10-29。环境：Python 3.11、PyTorch `2.7.0+cpu`，脚本默认配置、单线程 CPU，无外部数据或模型下载。

## 复现命令

在仓库根目录使用相同环境运行：

```bash
python exercises/stage1_pytorch.py --device cpu --epochs 20 --out outputs/continuous
python exercises/stage1_pytorch.py --device cpu --epochs 20 --stop-after 10 --out outputs/interrupted
python exercises/stage1_pytorch.py --device cpu --epochs 20 --resume outputs/interrupted/last.pt --out outputs/resumed
```

## 已验证结果

| 检查 | 结果 |
| --- | --- |
| 张量乘法、广播、标量梯度及梯度累积 | 断言通过 |
| 32 条样本过拟合 | 准确率 100%，参数发生更新 |
| 20 轮独立验证集准确率 | 约 98.0% |
| 第 20 轮验证 loss | 约 0.0526，低于初始值 |
| 新模型实例加载检查点 | 固定输入 logits 一致 |
| 第 10 轮中断后，在独立进程恢复到第 20 轮 | epoch=20、global_step=320 |
| 连续训练与恢复训练对照 | 模型参数逐项相等，完整指标历史一致 |

## 适用范围

这是合成分类数据上的教学验证，不代表 Qwen3 微调、GPU 环境或金融抽取任务已经通过。不同硬件、版本或更复杂训练的数值可能不同。

检查点与完整运行输出不随仓库发布，可按上述命令自行生成。阶段二至五尚处于学习计划阶段。
