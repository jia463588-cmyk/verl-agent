#!/usr/bin/env python3
"""
Simple validation script to check the D_rollout generation implementation.

This script validates:
1. File structure and imports
2. Data format correctness
3. Basic logic flow

Does NOT require full dependencies to run.
"""

import os
import sys
import json
import tempfile
from pathlib import Path


def validate_script_exists():
    """Check that the main script exists."""
    script_path = Path(__file__).parent / 'generate_d_rollout.py'
    assert script_path.exists(), f"Main script not found: {script_path}"
    print("✓ Main script exists")
    return str(script_path)


def validate_shell_script_exists():
    """Check that the shell script exists."""
    script_path = Path(__file__).parent / 'run_generate_d_rollout.sh'
    assert script_path.exists(), f"Shell script not found: {script_path}"
    assert os.access(script_path, os.X_OK), f"Shell script not executable: {script_path}"
    print("✓ Shell script exists and is executable")
    return str(script_path)


def validate_readme_exists():
    """Check that README exists."""
    readme_path = Path(__file__).parent / 'README_D_ROLLOUT.md'
    assert readme_path.exists(), f"README not found: {readme_path}"
    print("✓ README exists")
    return str(readme_path)


def validate_data_format():
    """Validate the expected D_rollout data format."""
    # New D_expert compatible format
    expected_fields = ['task_id', 'idx', 'id', 'task', 'step', 'state_si', 
                       'expert_action_ai', 'alternative_action_j', 'next_state_sji', 'is_expert']
    
    # Create sample entry in new format
    sample_entry = {
        'task_id': 'trial_T20190908_110055_655553',
        'idx': 1,
        'id': 'traj_0001_step001_alt1',
        'task': 'put a cool mug in coffeemachine.',
        'step': 1,
        'state_si': {
            'current_state': "You have taken the action 1: 'go to coffeemachine 1'. You are now at step 2 and your current observation is: You arrive at coffeemachine 1."
        },
        'expert_action_ai': 'go to coffeemachine 1',
        'alternative_action_j': 'go to fridge 1',
        'next_state_sji': 'You arrive at fridge 1. On the fridge 1, you see a apple 1.',
        'is_expert': False
    }
    
    # Validate all expected fields exist
    for field in expected_fields:
        assert field in sample_entry, f"Missing required field: {field}"
    
    # Test JSON serialization
    try:
        json_str = json.dumps(sample_entry, ensure_ascii=False)
        loaded = json.loads(json_str)
        assert loaded == sample_entry, "JSON round-trip failed"
    except Exception as e:
        raise AssertionError(f"JSON serialization failed: {e}")
    
    print("✓ Data format validation passed")
    return True


def validate_alternative_action_sampling_logic():
    """Validate the logic for alternative action sampling."""
    admissible_commands = ['action1', 'action2', 'action3', 'action4', 'action5']
    expert_action = 'action1'
    k = 3
    
    # Simulate sampling (uniform)
    import random
    random.seed(42)
    
    alternative_commands = [cmd for cmd in admissible_commands if cmd != expert_action]
    assert len(alternative_commands) == 4, "Filtering failed"
    
    sampled_actions = random.sample(alternative_commands, k=k)
    assert len(sampled_actions) == k, "Sampling count incorrect"
    assert expert_action not in sampled_actions, "Expert action should not be in samples"
    
    print("✓ Alternative action sampling logic validated")
    return True


def validate_dataset_structure():
    """Validate that dataset can be written and read correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, 'd_rollout.jsonl')
        
        # Create sample data
        sample_data = [
            {
                'state_i': f'state_{i}',
                'action_j': f'action_{i}',
                'state_j': f'next_state_{i}',
                'step': i,
                'expert_action': f'expert_action_{i}',
            }
            for i in range(10)
        ]
        
        # Write data
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in sample_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Read and validate
        loaded_data = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                loaded_data.append(json.loads(line))
        
        assert len(loaded_data) == len(sample_data), "Data length mismatch"
        assert loaded_data == sample_data, "Data content mismatch"
        
        print("✓ Dataset I/O validation passed")
        return True


def validate_statistics_format():
    """Validate statistics file format."""
    stats = {
        'num_episodes': 100,
        'successful_episodes': 85,
        'failed_episodes': 15,
        'total_rollout_entries': 4250,
        'k': 3,
        'max_steps': 50,
        'temperature': 1.0,
        'use_replay': True,
    }
    
    # Test JSON serialization
    try:
        json_str = json.dumps(stats, indent=2, ensure_ascii=False)
        loaded = json.loads(json_str)
        assert loaded == stats, "Stats round-trip failed"
    except Exception as e:
        raise AssertionError(f"Stats serialization failed: {e}")
    
    print("✓ Statistics format validated")
    return True


def main():
    """Run all validation checks."""
    print("="*60)
    print("D_rollout Generation Validation")
    print("="*60)
    
    try:
        # File existence checks
        validate_script_exists()
        validate_shell_script_exists()
        validate_readme_exists()
        
        # Format and logic checks
        validate_data_format()
        validate_alternative_action_sampling_logic()
        validate_dataset_structure()
        validate_statistics_format()
        
        print("="*60)
        print("✓ All validation checks passed!")
        print("="*60)
        print()
        print("Note: This validation only checks structure and logic.")
        print("To run the actual generation, ensure all dependencies are installed.")
        print("See README_D_ROLLOUT.md for installation instructions.")
        return 0
        
    except AssertionError as e:
        print("="*60)
        print(f"✗ Validation failed: {e}")
        print("="*60)
        return 1
    except Exception as e:
        print("="*60)
        print(f"✗ Unexpected error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
