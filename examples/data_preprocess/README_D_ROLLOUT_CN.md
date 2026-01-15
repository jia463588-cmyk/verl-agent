# D_rollout 数据集生成

本模块基于 GiGPO 论文方法，为 ALFWorld 环境生成 D_rollout 数据集。

## 概述

D_rollout 数据集生成支持两种模式：

### 模式1：从预收集的专家轨迹生成（推荐）

如果您已经有 JSON 格式的专家轨迹数据（D_expert）：

1. **加载专家轨迹**：从 JSON 文件加载预收集的专家轨迹
2. **采样替代动作**：为专家轨迹中的每个状态 s_i 采样 K=3 个不同于专家动作的替代动作
3. **执行动作**：执行每个替代动作以观察产生的状态 s_j
4. **存储 Rollout 数据**：存储所有状态和动作的元组 (s_i, a_j, s_j)

### 模式2：从零开始收集专家轨迹

如果您还没有专家数据：

1. **收集专家轨迹**：运行专家智能体收集成功的轨迹
2. **采样替代动作**：为每个状态 s_i 采样 K=3 个替代动作
3. **执行动作**：执行每个替代动作以观察产生的状态 s_j
4. **存储 Rollout 数据**：存储元组 (s_i, a_j, s_j)

### 数据集格式

输出的 D_rollout 数据集采用与 D_expert 兼容的格式：
```json
{
  "task_id": "trial_T20190908_110055_655553",
  "idx": 1,
  "id": "traj_0001_step001_alt1",
  "task": "put a cool mug in coffeemachine.",
  "step": 1,
  "state_si": {
    "current_state": "You have taken the action 1: 'go to coffeemachine 1', action 2: 'take mug 1 from coffeemachine 1' You are now at step 3 and your current observation is: You pick up the mug 1 from the coffeemachine 1."
  },
  "expert_action_ai": "go to coffeemachine 1",
  "alternative_action_j": "go to fridge 1",
  "next_state_sji": "You arrive at fridge 1. On the fridge 1, you see a apple 1, a bowl 2, a bowl 1, a egg 1, a lettuce 1, a mug 2, a potato 2, and a potato 1.",
  "is_expert": false
}
```

字段说明：
- `task_id`: 任务/轨迹的唯一标识符
- `idx`: 轨迹索引（轨迹编号）
- `id`: 该条目的唯一标识符（格式：traj_XXXX_stepXXX_altX）
- `task`: 任务描述/目标
- `step`: 轨迹中的步骤编号（从1开始）
- `state_si.current_state`: 完整状态，包含动作历史和当前观察
- `expert_action_ai`: 专家智能体选择的动作
- `alternative_action_j`: 采样的替代动作（不同于专家动作）
- `next_state_sji`: 执行 alternative_action_j 后的结果状态
- `is_expert`: 布尔标志（替代动作为 false，专家动作为 true）

## 使用方法

### 模式1：使用预收集的专家数据（推荐）

如果您在 `dexpert_test.json` 中有专家轨迹数据：

```bash
cd /home/runner/work/verl-agent/verl-agent
bash examples/data_preprocess/run_generate_d_rollout.sh dexpert_test.json 3 1.0 data/d_rollout
```

这将：
- 从 `dexpert_test.json` 加载专家轨迹
- 为每个状态采样 3 个替代动作
- 使用温度 1.0 进行采样
- 将输出保存到 `data/d_rollout/`

### 模式2：从零开始收集专家轨迹

使用默认参数运行生成脚本：

```bash
cd /home/runner/work/verl-agent/verl-agent
bash examples/data_preprocess/run_generate_d_rollout.sh 100 3 50 1.0 data/d_rollout
```

这将：
- 收集 100 条专家轨迹
- 为每个状态采样 3 个替代动作
- 使用温度 1.0 进行采样
- 将输出保存到 `data/d_rollout/`

### 自定义参数

```bash
bash examples/data_preprocess/run_generate_d_rollout.sh NUM_EPISODES K MAX_STEPS TEMPERATURE OUTPUT_DIR
```

示例：
```bash
bash examples/data_preprocess/run_generate_d_rollout.sh 200 3 50 1.0 data/my_rollout
```

### 高级用法

直接使用 Python 脚本以获得更多控制：

**使用预收集的专家数据：**
```bash
python3 -m examples.data_preprocess.generate_d_rollout \
    --expert_file dexpert_test.json \
    --k 3 \
    --temperature 1.0 \
    --output_dir data/d_rollout \
    --log_level INFO
```

**从零开始收集专家轨迹：**
```bash
python3 -m examples.data_preprocess.generate_d_rollout \
    --num_episodes 100 \
    --k 3 \
    --max_steps 50 \
    --temperature 1.0 \
    --output_dir data/d_rollout \
    --use_replay \
    --log_level INFO
```

### 参数说明

- `--expert_file`: 专家轨迹 JSON 文件的路径（如果提供，将加载专家数据而不是收集）
- `--num_episodes`: 要收集的专家轨迹数量（默认：100，仅在未提供 --expert_file 时使用）
- `--k`: 每个状态采样的替代动作数量（默认：3）
- `--max_steps`: 每个轨迹的最大步数（默认：50，仅在收集轨迹时使用）
- `--temperature`: 替代动作采样的温度（默认：1.0）
- `--output_dir`: D_rollout 数据集的输出目录（默认：data/d_rollout）
- `--use_replay`: 使用重放方法执行动作并获取实际结果状态（仅在收集轨迹时使用）
- `--seed`: 随机种子以确保可重复性（默认：42）
- `--log_level`: 日志级别（默认：INFO）

## 实现细节

### 两种操作模式

**模式1：从预收集的专家数据（推荐）**
- 从 JSON 文件加载专家轨迹
- 无需运行专家智能体
- 更快更高效
- 当您已有 D_expert 数据时使用

**模式2：从零开始收集**
- 使用 ALFWorld 的手工编码专家智能体
- 实时收集专家轨迹
- 较慢但自包含
- 从头开始时使用

### 专家智能体（仅模式2）

实现使用 ALFWorld 的手工编码专家智能体，可以高成功率解决任务：
- `PickAndPlaceSimpleTWPolicy`
- `PickTwoObjAndPlaceTWPolicy`
- `LookAtObjInLightTWPolicy`
- `PickHeatThenPlaceInRecepTWPolicy`
- `PickCoolThenPlaceInRecepTWPolicy`
- `PickCleanThenPlaceInRecepTWPolicy`

### 替代动作采样

当前实现：从可执行命令中均匀采样（排除专家动作）

采样过程：
- 从可执行命令列表中过滤掉专家动作
- 从剩余可执行命令中均匀采样 K 个动作
- 如果替代选项少于 K 个，则进行带放回采样

**未来增强**：使用 LLM 的基于模型的采样，温度=1.0
- 在 `AlternativeActionSampler` 中提供了扩展点
- 包含实现文档

### 状态转换方法

**重放方法（推荐）**：
- 重置环境
- 重放专家动作 0..i-1
- 在步骤 i 执行替代动作 j
- 观察结果状态 sj
- 复杂度：O(n²k)，其中 n=步数，k=替代动作数

**基本方法（已弃用）**：
- 仅存储元数据
- 不执行动作
- state_j 为 None
- 更快但不完整

## 输出文件

生成过程创建：

1. **d_rollout.jsonl**：主数据集文件，rollout 条目（每行一个 JSON）
2. **statistics.json**：关于生成过程的统计信息
3. **logs/generate_d_rollout_*.log**：详细执行日志

### 统计文件

```json
{
  "expert_file": "dexpert_test.json",
  "num_trajectories": 100,
  "total_rollout_entries": 4250,
  "k": 3,
  "temperature": 1.0
}
```

## 与 verl-agent 的集成

生成的 D_rollout 数据集可用于：

1. **训练**：转换为 parquet 格式并与 verl-agent 训练器一起使用
2. **分析**：分析状态-动作分布和转换动态
3. **评估**：比较替代动作结果与专家动作

### 转换为训练格式

```python
import pandas as pd
import json

# 加载 D_rollout
rollout_data = []
with open('data/d_rollout/d_rollout.jsonl', 'r') as f:
    for line in f:
        rollout_data.append(json.loads(line))

# 转换为 DataFrame 并保存为 parquet
df = pd.DataFrame(rollout_data)
df.to_parquet('data/d_rollout/d_rollout.parquet')
```

## 依赖要求

在运行 D_rollout 生成之前，请确保您已：

1. **安装 verl-agent**：
   ```bash
   cd /path/to/verl-agent
   pip install -e .
   ```

2. **安装 ALFWorld 环境**：
   ALFWorld 环境包含在仓库中：
   `agent_system/environments/env_package/alfworld/`
   
   安装其依赖：
   ```bash
   cd agent_system/environments/env_package/alfworld
   pip install -e .
   ```

3. **核心依赖**：
   - Python 3.8+
   - PyTorch
   - Ray（用于并行环境执行）
   - transformers
   - requirements.txt 中的其他依赖

### 快速安装

```bash
# 安装 verl-agent 和所有依赖
cd /path/to/verl-agent
pip install -e .
pip install -r requirements.txt

# 安装 ALFWorld
cd agent_system/environments/env_package/alfworld
pip install -e .
```

## 故障排除

### 专家智能体失败

如果专家智能体频繁失败：
- 检查任务类型是否正确识别
- 验证 ALFWorld 配置是否正确
- 查看日志中的特定失败模式

### 内存问题

对于大规模生成：
- 减少 `num_episodes` 并分批运行
- 分块处理数据
- 使用 `--use_replay False` 减少内存（尽管状态不准确）

### 环境问题

如果环境初始化失败：
- 检查 ALFWorld 安装
- 验证配置路径是否正确
- 确保 Ray 正确初始化

## 示例

### 生成小型测试数据集

```bash
python3 -m examples.data_preprocess.generate_d_rollout \
    --expert_file dexpert_test.json \
    --k 3 \
    --output_dir data/d_rollout_test
```

### 生成大型生产数据集

使用预收集的专家数据：
```bash
python3 -m examples.data_preprocess.generate_d_rollout \
    --expert_file dexpert_production.json \
    --k 3 \
    --output_dir data/d_rollout_production \
    --seed 42
```

从零开始收集：
```bash
python3 -m examples.data_preprocess.generate_d_rollout \
    --num_episodes 1000 \
    --k 3 \
    --max_steps 50 \
    --use_replay \
    --output_dir data/d_rollout_production \
    --seed 42
```

### 分析生成的数据集

生成后，分析数据集以了解其特征：

```bash
python3 examples/data_preprocess/analyze_d_rollout.py \
    data/d_rollout/d_rollout.jsonl \
    --output data/d_rollout/analysis_report.json
```

这将显示：
- 基本统计信息（总条目数、唯一状态/动作）
- 动作分布分析
- 结果比较
- 数据质量检查

## 参考文献

基于 GiGPO 论文中描述的方法，用于从专家轨迹生成多样化的 rollout 数据以改进智能体训练。

**引用：**
```bibtex
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```

**论文链接：** https://arxiv.org/abs/2505.10978

## 性能考虑

### 时间复杂度
重放方法的复杂度为 O(n²k)，其中：
- n = 轨迹长度（步数）
- k = 每个状态的替代动作数（默认：3）

对于 50 步轨迹，k=3 时，这导致约 3,750 个环境步骤。

### 优化策略
1. **限制轨迹长度**：使用 `--max_steps` 限制轨迹长度
2. **减少替代数量**：降低 k 值（例如，k=2）
3. **早期停止**：仅处理每条轨迹的前 N 步
4. **并行处理**：同时处理多个轨迹（未来增强）
5. **状态检查点**：如果环境支持（未来增强）

## 添加文件说明

### 1. generate_d_rollout.py（核心生成脚本，800+ 行）

**主要功能：**
- `ExpertAgent`: 专家智能体包装器，支持 ALFWorld 的 6 种任务类型
- `AlternativeActionSampler`: 替代动作采样器（当前为均匀采样，可扩展为基于模型）
- `collect_expert_trajectory()`: 使用专家智能体收集单条轨迹
- `load_expert_trajectories_from_file()`: 从 JSON 文件加载预收集的专家轨迹
- `generate_d_rollout_from_expert_file()`: 从预收集的专家数据生成 D_rollout
- `generate_d_rollout_with_replay()`: 通过重放轨迹生成 D_rollout
- `main()`: 主函数，支持两种模式（从文件加载或从零收集）

**工作流程：**
1. 解析命令行参数
2. 检测是否提供了 `--expert_file`
3. 如果提供：加载专家数据 → 为每个状态采样替代动作 → 执行并记录结果
4. 如果未提供：运行专家智能体收集轨迹 → 采样替代动作 → 执行并记录结果
5. 保存 D_rollout 数据集为 JSONL 格式

### 2. run_generate_d_rollout.sh（Shell 脚本包装器）

**功能：**
- 提供便捷的命令行接口
- 自动检测模式（基于参数）
- 支持两种使用方式：
  - 带文件路径：`bash run_generate_d_rollout.sh dexpert_test.json 3 1.0 output_dir`
  - 不带文件：`bash run_generate_d_rollout.sh 100 3 50 1.0 output_dir`

### 3. analyze_d_rollout.py（数据分析工具，244 行）

**功能：**
- 加载并分析 D_rollout 数据集
- 计算基本统计信息（条目总数、唯一状态/动作）
- 分析动作分布
- 检查数据质量
- 生成分析报告（可选保存为 JSON）

**兼容性：**
- 支持旧格式和新格式的数据
- 自动检测数据格式并相应调整分析

### 4. validate_d_rollout.py（验证脚本，199 行）

**功能：**
- 验证文件结构（检查主脚本、Shell 脚本、README 是否存在）
- 验证数据格式（测试 JSON 序列化、字段完整性）
- 验证采样逻辑
- 验证 I/O 操作
- 无需完整依赖即可运行

**使用场景：**
- 在安装完整依赖之前验证代码结构
- 快速检查实现是否符合规范

### 5. README_D_ROLLOUT.md（英文文档，308 行）

**内容：**
- 完整的使用说明
- 两种模式的详细说明
- 参数说明
- 实现细节
- 示例代码
- 故障排除指南

### 6. README_D_ROLLOUT_CN.md（中文文档，本文件）

**内容：**
- 所有英文 README 的内容翻译为中文
- 更适合中文用户阅读
- 包含相同的技术细节和使用说明
