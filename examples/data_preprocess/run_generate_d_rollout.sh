#!/bin/bash
# Script to generate D_rollout dataset from expert trajectories

set -e

# Check if expert file is provided as first argument
EXPERT_FILE=${1:-""}

if [ -n "$EXPERT_FILE" ]; then
    # Mode 1: Use existing expert trajectory file
    K=${2:-3}
    TEMPERATURE=${3:-1.0}
    OUTPUT_DIR=${4:-data/d_rollout}
    
    echo "Generating D_rollout dataset from expert file:"
    echo "  Expert file: $EXPERT_FILE"
    echo "  Alternative actions per state (K): $K"
    echo "  Sampling temperature: $TEMPERATURE"
    echo "  Output directory: $OUTPUT_DIR"
    
    python3 -m examples.data_preprocess.generate_d_rollout \
        --expert_file $EXPERT_FILE \
        --k $K \
        --temperature $TEMPERATURE \
        --output_dir $OUTPUT_DIR \
        --log_level INFO
else
    # Mode 2: Collect expert trajectories from scratch
    NUM_EPISODES=${1:-100}
    K=${2:-3}
    MAX_STEPS=${3:-50}
    TEMPERATURE=${4:-1.0}
    OUTPUT_DIR=${5:-data/d_rollout}
    
    # Use replay method to get actual resulting states
    USE_REPLAY="--use_replay"
    
    echo "Generating D_rollout dataset by collecting expert trajectories:"
    echo "  Number of episodes: $NUM_EPISODES"
    echo "  Alternative actions per state (K): $K"
    echo "  Max steps per episode: $MAX_STEPS"
    echo "  Sampling temperature: $TEMPERATURE"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Use replay method: Yes"
    
    python3 -m examples.data_preprocess.generate_d_rollout \
        --num_episodes $NUM_EPISODES \
        --k $K \
        --max_steps $MAX_STEPS \
        --temperature $TEMPERATURE \
        --output_dir $OUTPUT_DIR \
        $USE_REPLAY \
        --log_level INFO
fi

echo ""
echo "D_rollout generation complete!"
echo "Check output at: $OUTPUT_DIR"

