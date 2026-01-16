# D_rollout 数据集生成工具

本模块基于 GiGPO 论文方法，从预收集的专家轨迹（D_expert）生成 D_rollout 数据集用于 ALFWorld 环境。

## 概述

D_rollout 数据集生成流程：

1. **加载专家轨迹**：从 JSON 文件（如 `dexpert_test.json`）加载预收集的专家轨迹
2. **采样替代动作**：为专家轨迹中的每个状态 s_i 采样 K 个不同于专家动作的替代动作
3. **执行动作**：在 ALFWorld 环境中执行每个替代动作以观察产生的状态 s_j
4. **存储 Rollout 数据**：存储所有状态转移元组 (s_i, a_j, s_j)

### 数据集格式

输出的 D_rollout 数据集采用与 D_expert 兼容的 JSONL 格式：

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

**字段说明：**
- `task_id`: 任务/轨迹的唯一标识符
- `idx`: 轨迹索引（轨迹编号）
- `id`: 该条目的唯一标识符（格式：traj_XXXX_stepXXX_altX）
- `task`: 任务描述/目标
- `step`: 轨迹中的步骤编号（从1开始）
- `state_si.current_state`: 完整状态，包含动作历史和当前观察
- `expert_action_ai`: 专家智能体选择的动作
- `alternative_action_j`: 采样的替代动作（不同于专家动作）
- `next_state_sji`: 执行 alternative_action_j 后的结果状态
- `is_expert`: 布尔标志（替代动作为 false）

## 使用方法

### 基本用法

```bash
# 切换到项目根目录
cd /home/runner/work/verl-agent/verl-agent

# 使用默认参数生成 D_rollout
bash agent_system/environments/env_package/alfworld/data_generation/D_rollout/run_generate_d_rollout.sh \
    agent_system/environments/env_package/alfworld/data_generation/D_rollout/dexpert_test.json
```

### 自定义参数

```bash
bash agent_system/environments/env_package/alfworld/data_generation/D_rollout/run_generate_d_rollout.sh \
    <专家文件> [K] [温度] [输出目录] [模型路径]
```

**参数说明：**
- `<专家文件>`（必需）：专家轨迹 JSON 文件路径
- `[K]`（可选，默认3）：每个状态采样的替代动作数量
- `[温度]`（可选，默认1.0）：采样温度
- `[输出目录]`（可选，默认data/d_rollout）：输出目录
- `[模型路径]`（可选）：离线模型本地路径（如果提供则使用模型采样）

**示例：**

```bash
# 示例1：使用默认参数
bash agent_system/environments/env_package/alfworld/data_generation/D_rollout/run_generate_d_rollout.sh \
    agent_system/environments/env_package/alfworld/data_generation/D_rollout/dexpert_test.json

# 示例2：自定义 K 和输出目录
bash agent_system/environments/env_package/alfworld/data_generation/D_rollout/run_generate_d_rollout.sh \
    agent_system/environments/env_package/alfworld/data_generation/D_rollout/dexpert_test.json 5 1.0 my_output

# 示例3：使用离线模型
bash agent_system/environments/env_package/alfworld/data_generation/D_rollout/run_generate_d_rollout.sh \
    agent_system/environments/env_package/alfworld/data_generation/D_rollout/dexpert_test.json 3 1.0 data/d_rollout /path/to/local/model
```

### 使用 Python 直接调用

```bash
python3 -m agent_system.environments.env_package.alfworld.data_generation.D_rollout.generate_d_rollout \
    --expert_file agent_system/environments/env_package/alfworld/data_generation/D_rollout/dexpert_test.json \
    --k 3 \
    --temperature 1.0 \
    --output_dir data/d_rollout
```

### 使用离线模型进行动作采样

本工具支持从本地路径加载离线模型进行智能动作采样：

```bash
python3 -m agent_system.environments.env_package.alfworld.data_generation.D_rollout.generate_d_rollout \
    --expert_file agent_system/environments/env_package/alfworld/data_generation/D_rollout/dexpert_test.json \
    --k 3 \
    --use_model \
    --model_path /path/to/your/local/model \
    --output_dir data/d_rollout
```

**模型要求：**
- 模型必须保存在本地路径（离线模型）
- 支持 Hugging Face transformers 格式
- 包含模型权重和分词器文件
- 自动设置 `local_files_only=True` 和 `trust_remote_code=True`

**注意：** 如果不提供 `--model_path` 参数，系统将使用默认的均匀采样方法。

## 输出文件

生成的文件将保存在指定的输出目录中：

- `d_rollout.jsonl`: 主数据文件，包含所有 (s_i, a_j, s_j) 元组
- `d_rollout_stats.json`: 统计信息文件

## 数据分析

使用分析工具查看生成的数据集统计信息：

```bash
python3 agent_system/environments/env_package/alfworld/data_generation/D_rollout/analyze_d_rollout.py \
    data/d_rollout/d_rollout.jsonl
```

## 数据验证

验证生成的数据集格式：

```bash
python3 agent_system/environments/env_package/alfworld/data_generation/D_rollout/validate_d_rollout.py \
    data/d_rollout/d_rollout.jsonl
```

## 工具文件说明

1. **generate_d_rollout.py**（主生成脚本）
   - 从专家轨迹文件加载数据
   - 采样替代动作（均匀采样或模型采样）
   - 在 ALFWorld 环境中执行动作
   - 生成 D_rollout 数据集

2. **run_generate_d_rollout.sh**（Shell 包装脚本）
   - 简化命令行调用
   - 自动处理参数

3. **analyze_d_rollout.py**（数据分析工具）
   - 统计信息分析
   - 动作分布分析
   - 数据质量检查

4. **validate_d_rollout.py**（验证脚本）
   - 验证数据格式
   - 检查数据完整性

5. **dexpert_test.json**（示例专家轨迹数据）
   - 包含两个任务的专家轨迹
   - 用于测试和演示

## 常见问题

### 1. 如何准备专家轨迹文件？

专家轨迹文件应该是 JSON 格式，包含以下字段：
- `task_id`: 任务标识符
- `idx`: 轨迹索引
- `step`: 步骤编号
- `state_si.current_state`: 当前状态
- `expert_action_ai`: 专家动作
- `task`: 任务描述

参考示例文件 `dexpert_test.json`。

### 2. 生成的数据集用于什么？

D_rollout 数据集用于：
- 离线强化学习训练
- 反事实经验学习
- 探索替代动作的结果
- 提高智能体决策质量

### 3. 如何调整采样策略？

- 增加 K 值可以采样更多替代动作
- 调整温度参数影响采样随机性
- 使用 `--use_model` 和 `--model_path` 启用模型采样

## 技术细节

- **实现方法**：轨迹重放（Trajectory Replay）
- **复杂度**：O(n²k)，其中 n 是专家轨迹长度，k 是替代动作数
- **采样方式**：
  - 默认：从可执行命令中均匀采样
  - 可选：使用离线模型智能采样
- **环境**：ALFWorld（基于 TextWorld 的交互式环境）

## 参考文献

基于 GiGPO 论文中的 D_rollout 数据集生成方法。

## 许可证

Apache License 2.0
