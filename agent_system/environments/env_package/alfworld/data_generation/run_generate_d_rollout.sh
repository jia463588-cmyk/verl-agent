#!/bin/bash
# 从专家轨迹生成 D_rollout 数据集的脚本

set -e

# 检查第一个参数是否为专家文件
EXPERT_FILE=${1:-""}

if [ -n "$EXPERT_FILE" ]; then
    # 模式1：使用现有专家轨迹文件
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
    CMD="python3 -m agent_system.environments.env_package.alfworld.data_generation.generate_d_rollout \
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
else
    # 模式2：从零开始收集专家轨迹
    NUM_EPISODES=${1:-100}
    K=${2:-3}
    MAX_STEPS=${3:-50}
    TEMPERATURE=${4:-1.0}
    OUTPUT_DIR=${5:-data/d_rollout}
    
    # 使用重放方法获取实际结果状态
    USE_REPLAY="--use_replay"
    
    echo "通过收集专家轨迹生成 D_rollout 数据集："
    echo "  轨迹数量: $NUM_EPISODES"
    echo "  每个状态的替代动作数 (K): $K"
    echo "  每个轨迹的最大步数: $MAX_STEPS"
    echo "  采样温度: $TEMPERATURE"
    echo "  输出目录: $OUTPUT_DIR"
    echo "  使用重放方法: 是"
    
    python3 -m agent_system.environments.env_package.alfworld.data_generation.generate_d_rollout \
        --num_episodes $NUM_EPISODES \
        --k $K \
        --max_steps $MAX_STEPS \
        --temperature $TEMPERATURE \
        --output_dir $OUTPUT_DIR \
        $USE_REPLAY \
        --log_level INFO
fi

echo ""
echo "D_rollout 生成完成！"
echo "查看输出: $OUTPUT_DIR"

