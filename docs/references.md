# 参考项目与内容边界

本仓库以 smol-course 的学习结构为实战参考，加入基础训练练习，并计划将任务适配到 Qwen3-1.7B。不是官方课程镜像，也没有整体搬运上游教程或 Notebook。

| 来源 | 本项目中的用途 |
| --- | --- |
| [smol-course](https://github.com/huggingface/smol-course) | 指令微调、聊天模板、SFT 与 PEFT 的阅读顺序及练习主题 |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/zh-CN/chapter1/1) | Transformers、Datasets 与基础微调概念 |
| [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | 张量、自动求导、训练循环与保存加载 API |
| [Happy-LLM](https://github.com/datawhalechina/happy-llm) | 中文 Transformer 原理、手写模型与训练流程参考 |
| [TRL](https://github.com/huggingface/trl)、[PEFT](https://github.com/huggingface/peft) | 后续 SFT 与 LoRA 代码/API 参考 |
| [Qwen3](https://github.com/QwenLM/Qwen3) | 模型机制、聊天模板与使用方式 |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) | 主线完成后的配置化工具对照 |

仓库中的二维分类样本由脚本生成，不含实际金融业务数据。文档中的公司名与金额例子是虚构教学示例。

当前阶段一代码是本项目独立组织的教学实现，未包含整段上游源码文件。以后若复制或修改上游代码、文档或数据，应在对应文件标明来源与改动，并保留其许可要求的声明。外部内容不因被引用而改用本项目的 MIT 许可。

引用章节随上游变化，阅读时应同时核对标题、仓库修订版本和依赖版本。smol-course 实际章节映射与 README 可能存在差异，当前核对结果记录在[学习路线](learning-roadmap.md)。
