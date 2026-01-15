# D_rollout Dataset Generation

This module generates D_rollout datasets from expert trajectories for ALFWorld environments, based on the GiGPO paper methodology.

## Overview

The D_rollout dataset generation process:

1. **Collect Expert Trajectories (D_expert)**: Run expert agents to collect successful trajectories
2. **Sample Alternative Actions**: For each state s_i in expert trajectories, sample K=3 alternative actions different from the expert action
3. **Execute Actions**: Execute each alternative action to observe the resulting state s_j
4. **Store Rollout Data**: Store tuples (s_i, a_j, s_j) for all states and alternative actions

### Dataset Format

The output D_rollout dataset contains entries in the D_expert compatible format:
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

Where:
- `task_id`: Unique identifier for the episode/task
- `idx`: Trajectory index (episode number)
- `id`: Unique identifier for this entry (format: traj_XXXX_stepXXX_altX)
- `task`: Task description/goal
- `step`: Step number in the trajectory (1-indexed)
- `state_si.current_state`: Full state with action history and current observation
- `expert_action_ai`: The action the expert agent chose
- `alternative_action_j`: An alternative action sampled (different from expert action)
- `next_state_sji`: The resulting state after executing alternative_action_j
- `is_expert`: Boolean flag (false for alternative actions, would be true for expert actions)

## Usage

### Quick Start

Run the generation script with default parameters:

```bash
cd /home/runner/work/verl-agent/verl-agent
bash examples/data_preprocess/run_generate_d_rollout.sh
```

This will:
- Collect 100 expert trajectories
- Sample 3 alternative actions per state
- Use temperature 1.0 for sampling
- Save output to `data/d_rollout/`

### Custom Parameters

```bash
bash examples/data_preprocess/run_generate_d_rollout.sh NUM_EPISODES K MAX_STEPS TEMPERATURE OUTPUT_DIR
```

Example:
```bash
bash examples/data_preprocess/run_generate_d_rollout.sh 200 3 50 1.0 data/my_rollout
```

### Advanced Usage

Use the Python script directly for more control:

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

### Parameters

- `--num_episodes`: Number of expert episodes to collect (default: 100)
- `--k`: Number of alternative actions to sample per state (default: 3)
- `--max_steps`: Maximum steps per episode (default: 50)
- `--temperature`: Sampling temperature for alternative actions (default: 1.0)
- `--output_dir`: Output directory for D_rollout dataset (default: data/d_rollout)
- `--use_replay`: Use replay method to execute actions and get actual resulting states (recommended)
- `--seed`: Random seed for reproducibility (default: 42)
- `--log_level`: Logging level (default: INFO)

## Implementation Details

### Expert Agent

The implementation uses handcoded expert agents from ALFWorld that can solve tasks with high success rates:
- `PickAndPlaceSimpleTWPolicy`
- `PickTwoObjAndPlaceTWPolicy`
- `LookAtObjInLightTWPolicy`
- `PickHeatThenPlaceInRecepTWPolicy`
- `PickCoolThenPlaceInRecepTWPolicy`
- `PickCleanThenPlaceInRecepTWPolicy`

### Alternative Action Sampling

Currently implements uniform sampling from admissible commands (actions valid in the current state). The sampling:
- Filters out the expert action to ensure alternatives are different
- Samples K actions uniformly from remaining admissible commands
- Falls back to sampling with replacement if fewer than K alternatives exist

**Future Enhancement**: Model-based sampling using LLM inference with temperature=1.0 can be added by:
1. Loading a pretrained model
2. Generating action text with specified temperature
3. Validating against admissible commands

### Replay Method

The `--use_replay` option enables trajectory replay to get actual resulting states:
1. For each step i in expert trajectory:
   - Reset environment
   - Replay expert actions 0..i-1
   - Execute alternative action j
   - Observe resulting state s_j

This ensures accurate state transitions but requires more computation.

## Output Files

The generation process creates:

1. **d_rollout.jsonl**: Main dataset file with rollout entries (one JSON per line)
2. **statistics.json**: Statistics about the generation process
3. **logs/generate_d_rollout_*.log**: Detailed execution log

### Statistics File

```json
{
  "num_episodes": 100,
  "successful_episodes": 85,
  "failed_episodes": 15,
  "total_rollout_entries": 4250,
  "k": 3,
  "max_steps": 50,
  "temperature": 1.0,
  "use_replay": true
}
```

## Integration with verl-agent

The generated D_rollout dataset can be used:

1. **For Training**: Convert to parquet format and use with verl-agent trainers
2. **For Analysis**: Analyze state-action distributions and transition dynamics
3. **For Evaluation**: Compare alternative action outcomes with expert actions

### Converting to Training Format

```python
import pandas as pd
import json

# Load D_rollout
rollout_data = []
with open('data/d_rollout/d_rollout.jsonl', 'r') as f:
    for line in f:
        rollout_data.append(json.loads(line))

# Convert to DataFrame and save as parquet
df = pd.DataFrame(rollout_data)
df.to_parquet('data/d_rollout/d_rollout.parquet')
```

## Requirements

Before running the D_rollout generation, ensure you have:

1. **Install verl-agent**:
   ```bash
   cd /path/to/verl-agent
   pip install -e .
   ```

2. **Install ALFWorld environment**:
   The ALFWorld environment is included in the repository under:
   `agent_system/environments/env_package/alfworld/`
   
   Install its dependencies:
   ```bash
   cd agent_system/environments/env_package/alfworld
   pip install -e .
   ```

3. **Core dependencies**:
   - Python 3.8+
   - PyTorch
   - Ray (for parallel environment execution)
   - transformers
   - Other dependencies from requirements.txt

### Quick Installation

```bash
# Install verl-agent and all dependencies
cd /path/to/verl-agent
pip install -e .
pip install -r requirements.txt

# Install ALFWorld
cd agent_system/environments/env_package/alfworld
pip install -e .
```

## Troubleshooting

### Expert Agent Failures

If expert agents fail frequently:
- Check task types are correctly identified
- Verify ALFWorld configuration is correct
- Review logs for specific failure patterns

### Memory Issues

For large-scale generation:
- Reduce `num_episodes` and run multiple batches
- Process data in chunks
- Use `--use_replay False` to reduce memory (though states won't be accurate)

### Environment Issues

If environment fails to initialize:
- Check ALFWorld installation
- Verify config path is correct
- Ensure Ray is properly initialized

## Examples

### Generate Small Test Dataset

```bash
python3 -m examples.data_preprocess.generate_d_rollout \
    --num_episodes 10 \
    --k 3 \
    --use_replay \
    --output_dir data/d_rollout_test
```

### Generate Large Production Dataset

```bash
python3 -m examples.data_preprocess.generate_d_rollout \
    --num_episodes 1000 \
    --k 3 \
    --max_steps 50 \
    --use_replay \
    --output_dir data/d_rollout_production \
    --seed 42
```

### Analyze Generated Dataset

After generation, analyze the dataset to understand its characteristics:

```bash
python3 examples/data_preprocess/analyze_d_rollout.py \
    data/d_rollout/d_rollout.jsonl \
    --output data/d_rollout/analysis_report.json
```

This will show:
- Basic statistics (total entries, unique states/actions)
- Action distribution analysis
- Outcome comparisons
- Data quality checks

## References

Based on the methodology described in the GiGPO paper for generating diverse rollout data from expert trajectories to improve agent training.

**Citation:**
```bibtex
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```

**Paper Link:** https://arxiv.org/abs/2505.10978

## Performance Considerations

### Time Complexity
The replay method has O(n²k) complexity where:
- n = trajectory length (number of steps)
- k = alternatives per state (default: 3)

For a 50-step trajectory with k=3, this results in approximately 3,750 environment steps.

### Optimization Strategies
1. **Limit trajectory length**: Use `--max_steps` to cap episode length
2. **Reduce alternatives**: Lower k value (e.g., k=2)
3. **Early stopping**: Only process first N steps of each trajectory
4. **Parallel processing**: Process multiple episodes simultaneously (future enhancement)
5. **State checkpointing**: If environment supports it (future enhancement)
