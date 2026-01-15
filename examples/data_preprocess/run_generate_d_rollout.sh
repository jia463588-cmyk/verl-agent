#!/bin/bash
# Script to generate D_rollout dataset from expert trajectories

set -e

# Parameters
NUM_EPISODES=${1:-100}
K=${2:-3}
MAX_STEPS=${3:-50}
TEMPERATURE=${4:-1.0}
OUTPUT_DIR=${5:-data/d_rollout}

# Use replay method to get actual resulting states
USE_REPLAY="--use_replay"

echo "Generating D_rollout dataset with parameters:"
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

echo ""
echo "D_rollout generation complete!"
echo "Check output at: $OUTPUT_DIR"
