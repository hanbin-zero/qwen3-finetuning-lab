# 大模型基础实战训练

目标：理解语言模型如何训练，独立完成一次有数据、有评测、可复现的小模型微调，并能解释效果与失败原因。

路线：**PyTorch 基础 → 极小语言模型 → Qwen3 LoRA → 金融文本抽取评测 → 推理与复现**。按阶段验收推进，不承诺固定天数或训练耗时。

## 1. 学习资源与范围

阶段一默认 CPU，无需模型权重。后续以单张 24GB GPU 为参考开展 Qwen3-1.7B LoRA 实验；具体兼容版本、峰值显存和运行效果必须实测。

- 本项目以 smol-course 为实战主线，补充 PyTorch 基础与中文说明。
- 当前已提供阶段一完整教程与脚本；阶段二至五是学习计划，尚未提供完整训练实现。
- 实验使用独立环境和输出目录，模型权重、私有数据、检查点与认证信息不提交到仓库。
- GPU 编号由运行者选择；设置可见设备后，进程内编号可能重新从 0 开始。

## 2. 模型与工具选择

### 模型：主线替换为 Qwen3-1.7B

将原来的 `Qwen2.5-1.5B-Instruct` 替换为 **`Qwen/Qwen3-1.7B`**。这是选定的教学基线，不宣称是当前最新型号；模型升级后的业务收益仍需实测。

| 模型 | 用途 | 选择理由与约束 |
| --- | --- | --- |
| `Qwen/Qwen3-0.6B` | 可选的快速验证 | 先验证数据、模板、训练与保存流程，主线顺利时可跳过 |
| **`Qwen/Qwen3-1.7B`** | **基础 LoRA 主线** | 小规模纯文本模型，适合单卡反复实验，先做短序列、小批量试跑 |
| `Qwen/Qwen3-4B-Instruct-2507` | 后续业务对照 | 仅支持非思考模式；普通 LoRA 或 QLoRA 由试跑峰值显存决定 |
| `Qwen/Qwen3.5-2B` | 可选的新架构对照 | 更新的模型，含视觉编码器及 Gated DeltaNet/注意力混合结构，留到基础链路稳定后研究 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 兼容性备用 | 仅在已有环境适配受限时考虑，不同时维护两套入门实验 |

魔搭入口：[Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B)、[Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B)、[Qwen3-4B-Instruct-2507](https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)、[Qwen3.5-2B](https://modelscope.cn/models/Qwen/Qwen3.5-2B)。

魔搭页面依赖动态渲染，模型机制以文末官方模型卡交叉核对。训练选官方可训练权重，不直接拿现有推理服务的 GGUF、AWQ 文件替代基座；QLoRA 的量化加载是单独的训练配置。

### Qwen3 的模板与思考模式

- Qwen3-0.6B、1.7B 支持思考与非思考模式。本任务直接输出 JSON，训练与推理均明确使用非思考设置，检查 `enable_thinking=False` 是否真正传入模板处理。
- Qwen3-4B-Instruct-2507 本身只支持非思考模式，不照搬旧型号的模式切换参数。
- 使用所选模型自带的 tokenizer 和 chat template。训练通常不追加推理用的 generation prompt；推理按官方示例构造待生成的 assistant 开头。
- 解码检查实际训练输入与监督 token，避免误把提示词、推理占位或重复特殊 token 当作答案。非思考模板仍可能包含空思考标记，应检查并固定处理规则。

### 工具：先固定一套

使用 **PyTorch + Transformers + Datasets + PEFT + TRL**。前两阶段手写关键代码，微调阶段使用框架；主线完成后再比较 LLaMA-Factory、ms-swift 等封装工具。

实施时确认模型所需版本与服务器现状，在独立环境固定经过验证的版本。此处不提供未经服务器验证的安装命令。

### 项目导航：以 smol-course 为实战主线

推荐顺序：**本地 PyTorch 练习 → Happy-LLM 原理与极小模型 → LLM Course 第 2、3、5 章 → smol-course Unit 1 → 自定义评测与离线交付**。章节按下面的阶段对应选读，不要求先读完所有项目。

| 项目 | 定位 | 推荐入口 |
| --- | --- | --- |
| [smol-course](https://github.com/huggingface/smol-course) | 实战主线，训练任务统一换成 Qwen3-1.7B 金融文本抽取 | Unit 1 的指令微调、模板、SFT、LoRA 和练习 |
| [LLM Course 中文版](https://huggingface.co/learn/llm-course/zh-CN/chapter1/1) | Transformers 与 Datasets 基础教材 | [第 2 章：使用 Transformers](https://huggingface.co/learn/llm-course/zh-CN/chapter2/1)、[第 3 章：微调预训练模型](https://huggingface.co/learn/llm-course/zh-CN/chapter3/1)、[第 5 章：Datasets](https://huggingface.co/learn/llm-course/zh-CN/chapter5/1) |
| [Happy-LLM](https://github.com/datawhalechina/happy-llm) | 中文原理补充与手写模型参考 | 第 2 章 Transformer、第 5 章动手搭建、第 6 章训练实践；第 7 章的评测部分选读 |
| [TRL](https://github.com/huggingface/trl) | 训练 API 与代码参考 | [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)、[示例说明](https://huggingface.co/docs/trl/example_overview)，遇到参数问题查对应安装版本 |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) | 完成主线后的工程封装对照 | 将自己的数据和 LoRA 配置映射成工具配置，比较与 Python 脚本的差异 |

**章节核对说明（2026-10-29）**：smol-course README 与实际章节目录不完全一致。[当前目录](https://github.com/huggingface/smol-course/blob/main/units/en/_toctree.yml)中的 `unit2` 是偏好对齐/DPO，评测入口 `unit4/1` 标为 Coming soon，核对时对应源文件返回 404。因此不沿用“Unit 2 学评测”的旧推荐，不将该占位入口作为基础阶段依赖。章节位置变化时按标题核对。

smol-course Unit 1 按以下顺序阅读：

| 顺序 | 章节链接 | 本路线关注点 |
| --- | --- | --- |
| 1 | [1/1：Introduction to Instruction Tuning](https://github.com/huggingface/smol-course/blob/main/units/en/unit1/1.md) | 预训练、已有对话能力与任务微调的区别 |
| 2 | [1/2：Chat Templates](https://github.com/huggingface/smol-course/blob/main/units/en/unit1/2.md) | messages、角色、特殊 token、训练与生成的格式差别 |
| 3 | [1/3：Supervised Fine-Tuning](https://github.com/huggingface/smol-course/blob/main/units/en/unit1/3.md) | 数据准备、目标、训练配置、验证与日志 |
| 4 | [1/3a：LoRA and PEFT](https://github.com/huggingface/smol-course/blob/main/units/en/unit1/3a.md) | 冻结基座、低秩更新、适配器与显存开销 |
| 5 | [1/4：Hands-On Exercises](https://github.com/huggingface/smol-course/blob/main/units/en/unit1/4.md) | 依次完成模板、数据处理、SFT；CLI 练习留到交付阶段 |

课程示例以 SmolLM3 为主。本路线是借用其学习结构与练习目标，再适配 Qwen3；不将教材原始配置当成已经在单张训练 GPU 验证的配置。

### 各阶段对应阅读与练习

| 阶段 | 看哪些内容 | 做哪些练习 | 阶段出口 |
| --- | --- | --- | --- |
| 一：PyTorch 基础 | [阶段一教程](stage1-pytorch.md)；官方 PyTorch 基础教程 | 按教程 8 步做张量、梯度、Dataset、分类模型、训练/验证、小样本过拟合、保存与恢复；不加载 Qwen | 达到阶段一验收标准，再开始读微调代码 |
| 二：极小语言模型 | Happy-LLM 第 2 章、第 5 章；第 6 章的预训练部分选读 | 将教材结构缩小，手写 2～4 层 decoder-only 模型，检查因果遮罩与标签移位，训练前后生成对照 | 能解释 token 到 loss 的过程，不照搬教材完整模型规模 |
| 三：Qwen3 LoRA | 先读 LLM Course 第 2 章中的模型、Tokenizer、多序列处理，第 3 章的数据预处理、Trainer 与完整训练过程，第 5 章的本地数据、切片和创建数据集；再按上表读 smol-course Unit 1 | 改写 Unit 1/4 的 Exercise 1（模板）、Exercise 2（数据处理）、Exercise 3（SFTTrainer），使用 Qwen3-1.7B 和自己的抽取样本，显式加入 PEFT LoRA | 保存适配器，监督 token 检查通过；只完成短试跑，不宣称业务收益 |
| 四：独立评测 | 回看 smol-course Unit 1/3 的验证与监控；Happy-LLM 第 7 章评测部分选读；以本文第 6 节的评分规则为准 | 从 Exercise 3 延伸出独立评测脚本，对比固定提示词、少量示例和 LoRA；验证集调参，测试集最终验收，分析失败样本 | 交付三组逐条输出和指标，允许得出微调无收益的结论 |
| 五：推理与复现 | smol-course Unit 1/3a 的适配器部分，Unit 1/4 的 Exercise 4（TRL CLI，可选），TRL 当前版本的保存/加载说明 | 新进程加载同版基座、tokenizer、adapter；复现离线结果和资源开销；可选把 Python 配置映射到 CLI | 他人可按说明复现；不以发布到 Hub 或常驻部署作为必做项 |

阶段三内部按以下小步推进，不直接运行完整课程 Notebook：

1. **Transformers / Exercise 1**：用 3 条中文消息检查 Qwen3 的角色和特殊 token，比较生成提示与完整训练样本；在一条独立样本上理解思考开关，抽取主线固定非思考模式。
2. **Datasets / Exercise 2**：从本地 JSONL 读取 32～64 条人工核对样本，统一为 messages 或 prompt-completion 格式；实践 `map`、`filter`、`select`，统计 token 长度。正式划分按来源文档分组，不用随机切行替代防泄漏检查。
3. **PEFT / Unit 1/3a**：只挂载 LoRA 并检查参数名、参数量与冻结状态，先不追求训练效果。
4. **TRL / Exercise 3**：使用本文阶段三起步参数短试跑，检查实际监督 token、loss、峰值显存和保存加载，再扩大训练数据。
5. **评测与交付**：回到阶段四、五，不重复训练一套 SmolLM3，也不为了完成课程而先购买云服务或发布模型。

### 将课程适配到 Qwen3-1.7B 的检查表

- **模型**：固定官方 `Qwen/Qwen3-1.7B`（[魔搭](https://modelscope.cn/models/Qwen/Qwen3-1.7B)、[官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)），不改成 `-Base`，不自行增加 `-Instruct`。已有后训练模型上做任务 SFT，与教材从 Base 获得对话能力的目标不同。
- **本地路径**：模型目录由运行者指定，例如 `/path/to/models/Qwen3-1.7B`；先核对下载完整性和修订版本，再让模型和 tokenizer 指向同一目录。目录存在不代表可用。
- **模板与思考**：使用 Qwen3 自带模板和适配该版本的非思考配置，不复制 SmolLM3 的思考标记或自造特殊 token。检查模板结果，避免重复套模板。
- **数据与显存**：Exercise 2/3 改用自己的数据和短序列、小批量 LoRA；不因课程里存在全量数据加载、全参训练或较大 batch 就直接照搬。起步配置以本文阶段三为准。
- **API 与依赖**：课程片段可能使用旧参数或 collator；按锁定的 Transformers/PEFT/TRL 版本查官方文档，尤其检查监督掩码。当前 TRL 基础训练脚本位于 `trl/scripts`，不要硬找旧文章中的 `examples/scripts/sft.py`。
- **执行边界**：阅读课程不自动执行安装、登录、HF Jobs、上传或部署；实际训练前先准备独立环境和可用 GPU。

基础阶段暂不进入 smol-course 的 DPO、VLM、强化学习或合成数据大规模生产。LLaMA-Factory 放到同一任务的 Python 脚本、评测和适配器交付都完成后，用来理解工程封装。

## 3. 阶段一：PyTorch 与训练循环

详细教程：[阶段一：PyTorch 基础学习与练习](stage1-pytorch.md)。包含按顺序执行的 8 个步骤、示例代码和验收方法；完整脚本见 [stage1_pytorch.py](../exercises/stage1_pytorch.py)。本阶段 CPU 即可完成，不需要下载 Qwen 或修改服务器环境。

**学习内容**

- 张量形状、矩阵乘法、广播、数据类型与设备。
- Dataset、DataLoader、batch、前向计算、交叉熵、反向传播与优化器。
- `train()` / `eval()`、关闭梯度、梯度清零、学习率和随机种子。
- 训练集与验证集、过拟合、数据泄漏。

**实操任务**

1. 用小型分类模型完成“读取数据 → 训练 → 验证 → 保存 → 重新加载”。
2. 取固定 16～32 条样本刻意过拟合，关闭 dropout 等干扰，观察 loss 和预测变化。
3. 打印一个 batch 的形状、一次参数梯度和更新前后的数值。
4. 保存模型、优化器、训练步数及随机状态，演示中断后继续训练。

**验收标准**

- 能解释一次参数更新的步骤及为什么要清空梯度。
- 固定小批样本达到至少 95% 训练准确率；明确这只是训练链路自检，不代表泛化能力。
- 能分别报告训练和验证指标，解释“训练变好、验证变差”。
- 重新加载后固定输入的预测一致；恢复训练能继续更新参数，不要求跨硬件逐位一致。

**产物**：训练脚本、配置、训练/验证曲线及恢复说明。

## 4. 阶段二：从零训练极小语言模型

**学习内容**

- token、词表、embedding、自回归目标，以及 `[batch, seq, vocab]` logits 的意义。
- Q/K/V、缩放点积注意力、多头注意力、因果遮罩、残差、归一化与前馈网络。
- 位置编码：先实现简单位置 embedding，再理解 RoPE，不要求第一版复刻 Qwen。
- padding、attention mask、忽略位置、next-token 标签移位。

**实操任务**

1. 选择来源明确的小文本集，先按文档划分训练/验证集，再切片；第一版可用字符级 tokenizer。
2. 手写 2～4 层的小型 decoder-only Transformer，控制在百万到千万级参数，记录实际参数量。
3. 自己完成一个训练 step，打印输入、目标与 logits 形状，逐项检查预测位置。
4. 先在固定小批数据上过拟合，再正常训练；定期使用同一提示词生成样例。
5. 固定推理模式，改变序列后半部分，检查前半部分 logits 是否保持一致，验证因果遮罩。

**验收标准**

- 能画出 token 到 loss 的数据流，并解释主要维度。
- 自写 loss 只移位一次；以后使用框架内置 causal LM loss 时，先确认其内部处理，避免重复移位。
- 因果遮罩检查通过，padding 不计入目标 loss，固定小批样本可明显过拟合。
- 正常训练的验证 loss 相对初始检查点下降，并保留同一提示词的前后生成样例。
- 能区分“学到局部文本规律”和“具备可用问答能力”，不把 loss 下降等同于模型可用。

**产物**：模型源码、数据划分记录、loss 曲线及生成样例。规模同时记录文档数和 token 数，不按“几千条、几小时”保证效果。

## 5. 阶段三：Qwen3-1.7B LoRA 微调

**学习内容**

- Base 与后训练模型、预训练与 SFT、全量训练与参数高效微调。
- LoRA 低秩更新、冻结基座、rank、alpha、dropout 和目标模块。
- 普通 LoRA 与 QLoRA；量化基座仍有激活、梯度和训练状态开销。
- chat template、EOS、截断、监督掩码、梯度累积、混合精度及梯度检查点。

**实操任务**

1. 按下一阶段的抽取规则先整理 32～64 条人工核对样本，固定一批做训练自检，不动最终测试集。
2. 先用未微调的 Qwen3-1.7B 推理，保存原始输出作为基线。
3. 在单张训练 GPU 上用短序列、小批量普通 LoRA 试跑；如链路不通，可先换 0.6B 排除数据与模板问题。
4. 解码至少 3 条实际训练样本的输入和监督片段，覆盖正常、缺信息、长文本情况，检查答案未被截断，提示词和 padding 未误计入答案 loss。
5. 打印可训练参数名和数量，确认基座冻结、LoRA 参数有梯度；先在固定样本上过拟合，再扩展训练。
6. 保存 adapter，结束进程后重新加载“同一基座版本 + adapter + tokenizer”，复查输出。

**建议起点，仅供试跑调整**

| 配置 | 起点 | 调整原则 |
| --- | --- | --- |
| 模型 | Qwen3-1.7B | 全流程稳定前固定模型 |
| 序列总长度 | 512～1024 tokens，包含输入和答案 | 先统计长度，超长样本分段或单列，不静默截掉答案 |
| micro-batch | 1 | 先量峰值显存，再增大 |
| 梯度累积 | 8 | 单卡有效 batch = micro-batch × 累积步数 |
| LoRA rank / alpha | 8 / 16 | 首轮固定，后续单独对照 rank |
| 目标模块 | `q_proj`、`v_proj` | 核对实际模块名，再考虑扩大覆盖 |
| 学习率 | `1e-4` | 试验起点，结合验证集调整 |
| 精度 | 优先试 BF16 | 核实硬件与软件支持，不支持时单独设计 FP16 配置 |
| packing | 首轮关闭 | 先验证样本边界、监督与截断，再研究吞吐 |

TRL 的答案监督开关必须与数据格式、模板能力匹配。`assistant_only_loss` 依赖模板能提供 assistant mask，不能只开参数就认定正确；必要时使用受支持的 prompt-completion 格式或显式构造掩码，检查实际监督位置。

**验收标准**

- 一轮试跑无 OOM、无 NaN，记录峰值显存、步数、耗时及版本配置。
- 能解释哪些参数更新、哪些 token 参与 loss、有效 batch 如何计算。
- 固定小样本上明显过拟合，验证集与训练自检样本保持区分。
- adapter 保存和重新加载成功，保留禁用/启用 adapter 的输出对照。
- 业务效果由下一阶段判断，不以“训练结束”证明有效。

**产物**：训练配置、样本检查结果、adapter、训练记录与加载脚本。合并权重不是本阶段必做项。

## 6. 阶段四：金融文本抽取与独立评测

从公告或研报片段提取结构化信息。优先使用公开、来源明确的材料；模型生成的数据必须人工核对，测试集不以待评模型自评作为唯一答案。

### 输入和输出规则

第一版处理一个主要事件；多个事件的材料预先分段，多主体和含混表述作为专项样本。输出固定 JSON：

```json
{
  "主体": "甲公司",
  "事件类型": "回购",
  "日期": "2026-10-20",
  "金额": "1000",
  "金额单位": "万元",
  "证据原文": "甲公司于2026年10月20日回购股份，金额为1000万元。"
}
```

主体为主要事件主体；类型限定为“回购、分红、增持、减持、其他”；日期为事件日期，原文足以确定年月日才转为 `YYYY-MM-DD`，不得拿发布时间替代。金额用十进制字符串保留原文数值，单位单列。缺失字段填 `null`，信息不足以判断类型也填 `null`，不归入“其他”。证据必须是输入中的连续原文片段，不补充背景知识。

### 学习内容与实操任务

1. 学习去重、标注一致性、数据泄漏、验证集调参与测试集最终验收的区别。
2. 建立 500～1000 条训练样本、约 100 条验证样本和 100～200 条独立测试样本。先按来源文档分组切分，近重复材料归同组，保存清单。
3. 覆盖缺金额、无完整日期、多主体、易混事件、无相关事件；记录数量，至少抽取 50 条做第二遍独立核对。
4. 同一 Qwen3-1.7B 比较三组：固定提示词、提示词加固定少量示例、相同任务提示词加 LoRA。示例仅来自训练数据，固定解码配置与输出预算。
5. 用验证集做单变量实验，例如 rank 8 与 16，保持目标模块、数据、更新步数等设置一致，记录可训练参数变化，不同时改学习率。
6. 选定配置后做最终测试；若依据测试结果继续调参，这批数据已参与开发，需要新建独立测试集。

### 指标与验收标准

| 指标 | 计算约定 | 教学目标 |
| --- | --- | --- |
| 结构合法率 | 全部样本中能解析且键与类型正确的比例，不靠事后修复掩盖原始错误 | ≥95% |
| 非空字段 F1 | 比较主体、类型、日期、金额及单位的非空字段-值对；固定归一化规则，缺报计 FN、误报计 FP | 报告总体和各字段结果 |
| 缺信息误填率 | 标注应为 null 的字段中，输出非空值的比例 | ≤5% |
| 证据有效率 | 标注存在证据的样本中，输出连续原文且人工确认支持抽取结果的比例；缺失或无效 JSON 计失败 | 报告结果及失败类型 |
| 成本 | 同一输入集合上的峰值显存、端到端耗时及输入/输出 token 数 | 条件一致，可比较 |

以上数字是项目设定的教学目标，不是模型能力保证。无效 JSON 样本的字段按未抽取处理，另列结构错误；F1 不计双方均为空的字段，避免大量空值抬高分数。null 分母为零时记“不适用”，不写成 0%。

业务收益与两个未微调基线中较好的一个比较：预先约定主指标为非空字段 F1，目标提高至少 3 个百分点，同时缺信息误填率恶化不超过 1 个百分点。报告样本数及绝对差值；收益较小时，对候选配置至少重跑两个训练种子，不将小测试集的一次差值称为稳定提升。

未达收益目标时，完整报告负结果和原因也算完成学习验收，但模型不能通过业务采用验收。抽查至少 20 条失败样本，区分标注、模板/截断、格式、漏提取和无依据填值问题；不足 20 条则全部检查。

**产物**：数据及划分版本、评分脚本、逐条原始输出、三组对照表、失败分析及采用结论。

## 7. 阶段五：推理、复现与交付

**学习内容**

- tokenizer、基座、adapter 的版本绑定，评测模式与生成参数。
- 权重、激活、KV cache 和训练状态占用；吞吐与单请求时延。
- adapter 加载、权重合并、重新量化是不同步骤，训练与部署格式不能直接混用。

**实操任务**

1. 先用独立 Transformers 离线脚本推理，不以常驻服务部署作为验收前提。
2. 在新进程加载固定版本基座和 adapter，用同一测试输入与解码参数复现结果。
3. 固定输入集合，分别记录加载/预热时间、预热后推理耗时及峰值显存，注明长度、batch 和输出长度。
4. 整理最少运行说明、配置与依赖清单，让别人能找到数据、模型版本、训练入口、评测入口和结果。
5. 可选：在独立环境评估 vLLM 或合并导出；合并后重新评测，再量化后再次评测。另行确定服务端口与资源，不覆盖现有服务。

**验收标准**

- 新进程能加载并输出合法结果；固定版本、输入与推理条件的指标可复现，差异有记录。
- 另一位使用者能按说明完成离线评测，不依赖训练进程中的临时状态。
- 能用证据回答“微调是否更好、收益多大、代价多少、哪些样本仍会失败”。
- 合并、量化或部署未做时标为未验证，不据训练成功宣称线上可用。

**最终产物**：可复现实验材料及一页报告，包含任务、数据、基线、配置、指标、失败案例、资源开销和采用建议。

## 8. 主线完成后的进阶顺序

1. 单张训练 GPU 上评估 Qwen3-4B-Instruct-2507。先短序列试跑，显存不足再引入 QLoRA，将量化作为新变量记录。
2. 同一数据与评分规则比较 Qwen3.5-2B，学习新架构及加载/微调方式，不直接照搬 Qwen3 的模块配置。
3. 需要更大模型时安排 A100 独占窗口，选择经兼容性验证的 7B/8B 级模型做 LoRA 对照。
4. 最后学习多卡、分布式、长上下文与偏好优化。A100 和 3090 不视为自动相加的统一显存池，异构多卡另行设计和实测。

## 9. 官方参考资料

课程结构与模型机制于 2026-10-29 核对；运行实验时另记模型修订号与依赖版本。

- [PyTorch 基础训练流程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)：型号、思考模式及示例。
- [Qwen3-1.7B 官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)：主线模型与模板行为。
- [Qwen3-4B-Instruct-2507 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)：非思考版本。
- [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)：视觉编码器及混合架构。
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)：数据格式、监督范围和训练配置。
- [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora)、[量化训练](https://huggingface.co/docs/peft/developer_guides/quantization)：适配器与 QLoRA。
