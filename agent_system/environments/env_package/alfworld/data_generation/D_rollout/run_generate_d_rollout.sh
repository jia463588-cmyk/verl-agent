#!/bin/bash
# 从专家轨迹生成 D_rollout 数据集的脚本
# 
# 使用方法:
#   bash run_generate_d_rollout.sh <专家文件> [K] [温度] [输出目录] [模型路径]
#
# 参数:
#   专家文件: 必需，专家轨迹JSON文件路径（如 dexpert_test.json）
#   K: 可选，每个状态的替代动作数（默认: 3）
#   温度: 可选，采样温度（默认: 1.0）
#   输出目录: 可选，输出目录（默认: data/d_rollout）
#   模型路径: 可选，离线模型本地路径（如果提供则使用模型采样）

set -e

# 检查是否提供专家文件
EXPERT_FILE=${1}

if [ -z "$EXPERT_FILE" ]; then
    echo "错误: 必须提供专家轨迹文件路径"
    echo ""
    echo "使用方法:"
    echo "  bash run_generate_d_rollout.sh <专家文件> [K] [温度] [输出目录] [模型路径]"
    echo ""
    echo "示例:"
    echo "  bash run_generate_d_rollout.sh dexpert_test.json"
    echo "  bash run_generate_d_rollout.sh dexpert_test.json 3 1.0 data/d_rollout"
    echo "  bash run_generate_d_rollout.sh dexpert_test.json 3 1.0 data/d_rollout /path/to/model"
    exit 1
fi

# 设置参数默认值
K=${2:-3}
TEMPERATURE=${3:-1.0}
OUTPUT_DIR=${4:-data/d_rollout}
MODEL_PATH=${5:-""}

echo "从专家文件生成 D_rollout 数据集："
echo "  专家文件: $EXPERT_FILE"
echo "  每个状态的替代动作数 (K): $K"
echo "  采样温度: $TEMPERATURE"
echo "  输出目录: $OUTPUT_DIR"

# 构建命令
CMD="python3 -m agent_system.environments.env_package.alfworld.data_generation.D_rollout.generate_d_rollout \
    --expert_file $EXPERT_FILE \
    --k $K \
    --temperature $TEMPERATURE \
    --output_dir $OUTPUT_DIR \
    --log_level INFO"

# 如果提供了模型路径，添加模型相关参数
if [ -n "$MODEL_PATH" ]; then
    echo "  离线模型路径: $MODEL_PATH"
    CMD="$CMD --use_model --model_path $MODEL_PATH"
fi

# 执行命令
eval $CMD

echo ""
echo "D_rollout 生成完成！"
echo "查看输出: $OUTPUT_DIR"

