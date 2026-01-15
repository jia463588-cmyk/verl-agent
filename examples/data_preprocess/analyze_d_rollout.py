#!/usr/bin/env python3
"""
Utility script to analyze D_rollout datasets.

This script provides various analysis functions for D_rollout data:
- Basic statistics
- State-action distribution analysis
- Alternative action outcome comparison
- Data quality checks
"""

import os
import json
import argparse
from collections import defaultdict, Counter
from typing import List, Dict


def load_d_rollout_data(filepath: str) -> List[Dict]:
    """Load D_rollout dataset from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    return data


def compute_basic_statistics(data: List[Dict]) -> Dict:
    """Compute basic statistics about the D_rollout dataset."""
    # Handle both old and new format
    def get_state_key(entry):
        if 'state_si' in entry:
            return entry['state_si'].get('current_state', '')
        return entry.get('state_i', '')
    
    def get_action_key(entry):
        return entry.get('alternative_action_j', entry.get('action_j', ''))
    
    stats = {
        'total_entries': len(data),
        'unique_states': len(set(get_state_key(entry) for entry in data)),
        'unique_actions': len(set(get_action_key(entry) for entry in data)),
        'steps_distribution': Counter(entry['step'] for entry in data),
    }
    
    # Compute average alternatives per state
    states_count = defaultdict(int)
    for entry in data:
        states_count[get_state_key(entry)] += 1
    
    if states_count:
        stats['avg_alternatives_per_state'] = sum(states_count.values()) / len(states_count)
        stats['min_alternatives_per_state'] = min(states_count.values())
        stats['max_alternatives_per_state'] = max(states_count.values())
    
    # Add task-level statistics if available
    if any('task_id' in entry for entry in data):
        stats['unique_tasks'] = len(set(entry.get('task_id', '') for entry in data))
        stats['unique_trajectories'] = len(set(entry.get('idx', -1) for entry in data))
    
    return stats


def analyze_action_distribution(data: List[Dict]) -> Dict:
    """Analyze distribution of actions."""
    # Handle both old and new format
    def get_alt_action(entry):
        return entry.get('alternative_action_j', entry.get('action_j', ''))
    
    def get_expert_action(entry):
        return entry.get('expert_action_ai', entry.get('expert_action', ''))
    
    action_counts = Counter(get_alt_action(entry) for entry in data)
    expert_action_counts = Counter(get_expert_action(entry) for entry in data if get_expert_action(entry))
    
    return {
        'total_unique_alternative_actions': len(action_counts),
        'total_unique_expert_actions': len(expert_action_counts),
        'top_10_alternative_actions': action_counts.most_common(10),
        'top_10_expert_actions': expert_action_counts.most_common(10),
    }


def analyze_outcome_comparison(data: List[Dict]) -> Dict:
    """Analyze outcomes of alternative vs expert actions."""
    # New format doesn't have reward, check for next_state_sji existence
    has_next_state = any('next_state_sji' in entry for entry in data)
    
    if not has_next_state and not any('reward' in entry for entry in data):
        return {'note': 'Outcome information not available in dataset'}
    
    stats = {}
    
    # Check for reward-based metrics (old format)
    if any('reward' in entry for entry in data):
        alternative_rewards = [entry['reward'] for entry in data if 'reward' in entry]
        done_count = sum(1 for entry in data if entry.get('done', False))
        
        stats.update({
            'alternative_action_done_rate': done_count / len(data) if data else 0,
            'avg_alternative_reward': sum(alternative_rewards) / len(alternative_rewards) if alternative_rewards else 0,
            'max_alternative_reward': max(alternative_rewards) if alternative_rewards else 0,
            'min_alternative_reward': min(alternative_rewards) if alternative_rewards else 0,
        })
    
    # Add new format metrics
    if has_next_state:
        completed_transitions = sum(1 for entry in data if entry.get('next_state_sji', ''))
        stats['completed_transitions'] = completed_transitions
        stats['completion_rate'] = completed_transitions / len(data) if data else 0
    
    return stats


def analyze_step_distribution(data: List[Dict]) -> Dict:
    """Analyze distribution of data across trajectory steps."""
    step_data = defaultdict(list)
    for entry in data:
        step_data[entry['step']].append(entry)
    
    def get_state_key(entry):
        if 'state_si' in entry:
            return entry['state_si'].get('current_state', '')
        return entry.get('state_i', '')
    
    def get_action_key(entry):
        return entry.get('alternative_action_j', entry.get('action_j', ''))
    
    step_stats = {}
    for step, entries in step_data.items():
        step_stats[step] = {
            'count': len(entries),
            'unique_states': len(set(get_state_key(e) for e in entries)),
            'unique_actions': len(set(get_action_key(e) for e in entries)),
        }
    
    return {
        'total_steps': len(step_data),
        'entries_per_step': {k: v['count'] for k, v in step_stats.items()},
        'avg_entries_per_step': sum(v['count'] for v in step_stats.values()) / len(step_stats) if step_stats else 0,
    }


def check_data_quality(data: List[Dict]) -> Dict:
    """Check data quality and identify potential issues."""
    issues = []
    
    # Detect format (old vs new)
    is_new_format = any('task_id' in entry for entry in data)
    
    # Check for required fields based on format
    if is_new_format:
        required_fields = ['task_id', 'idx', 'id', 'task', 'step', 'state_si', 
                          'expert_action_ai', 'alternative_action_j', 'next_state_sji', 'is_expert']
    else:
        required_fields = ['state_i', 'action_j', 'state_j', 'step']
    
    for i, entry in enumerate(data):
        missing_fields = [f for f in required_fields if f not in entry]
        if missing_fields:
            issues.append(f"Entry {i}: Missing fields {missing_fields}")
    
    # Check for empty states or actions based on format
    if is_new_format:
        empty_states = sum(1 for entry in data if 'state_si' in entry and not entry['state_si'].get('current_state', '').strip())
        empty_actions = sum(1 for entry in data if not entry.get('alternative_action_j', '').strip())
        empty_next_states = sum(1 for entry in data if entry.get('next_state_sji') is not None and not entry['next_state_sji'].strip())
    else:
        empty_states = sum(1 for entry in data if not entry.get('state_i', '').strip())
        empty_actions = sum(1 for entry in data if not entry.get('action_j', '').strip())
        empty_next_states = sum(1 for entry in data if entry.get('state_j') is not None and not entry['state_j'].strip())
    
    # Check for duplicate entries
    entry_signatures = []
    for entry in data:
        if is_new_format:
            sig = (entry.get('id', ''), entry.get('task_id', ''))
        else:
            sig = (entry.get('state_i', ''), entry.get('action_j', ''), entry.get('step', -1))
        entry_signatures.append(sig)
    duplicate_count = len(entry_signatures) - len(set(entry_signatures))
    
    return {
        'total_issues': len(issues),
        'issue_details': issues[:10] if issues else [],  # Show first 10
        'empty_states': empty_states,
        'empty_actions': empty_actions,
        'empty_next_states': empty_next_states,
        'duplicate_entries': duplicate_count,
        'quality_score': 1.0 - (len(issues) + empty_states + empty_actions + duplicate_count) / max(len(data), 1),
    }


def print_analysis_report(data: List[Dict], output_file: str = None):
    """Generate and print comprehensive analysis report."""
    print("="*80)
    print("D_rollout Dataset Analysis Report")
    print("="*80)
    print()
    
    # Basic statistics
    print("BASIC STATISTICS")
    print("-"*80)
    basic_stats = compute_basic_statistics(data)
    for key, value in basic_stats.items():
        if key == 'steps_distribution':
            print(f"  {key}:")
            for step, count in sorted(value.items())[:10]:  # Show first 10 steps
                print(f"    Step {step}: {count} entries")
            if len(value) > 10:
                print(f"    ... ({len(value) - 10} more steps)")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Action distribution
    print("ACTION DISTRIBUTION")
    print("-"*80)
    action_stats = analyze_action_distribution(data)
    for key, value in action_stats.items():
        if 'top_10' in key:
            print(f"  {key}:")
            for action, count in value:
                print(f"    {action}: {count}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Outcome comparison
    print("OUTCOME ANALYSIS")
    print("-"*80)
    outcome_stats = analyze_outcome_comparison(data)
    for key, value in outcome_stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Step distribution
    print("STEP DISTRIBUTION")
    print("-"*80)
    step_stats = analyze_step_distribution(data)
    print(f"  Total steps: {step_stats['total_steps']}")
    print(f"  Avg entries per step: {step_stats['avg_entries_per_step']:.2f}")
    print(f"  Entries per step (first 10):")
    for step, count in sorted(step_stats['entries_per_step'].items())[:10]:
        print(f"    Step {step}: {count} entries")
    print()
    
    # Data quality
    print("DATA QUALITY CHECK")
    print("-"*80)
    quality_stats = check_data_quality(data)
    for key, value in quality_stats.items():
        if key == 'issue_details' and value:
            print(f"  {key} (showing first 10):")
            for issue in value:
                print(f"    {issue}")
        elif key != 'issue_details':
            print(f"  {key}: {value}")
    print()
    
    print("="*80)
    
    # Save report if output file specified
    if output_file:
        report = {
            'basic_statistics': basic_stats,
            'action_distribution': action_stats,
            'outcome_analysis': outcome_stats,
            'step_distribution': step_stats,
            'data_quality': quality_stats,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze D_rollout dataset')
    parser.add_argument('input_file', type=str, help='Path to D_rollout JSONL file')
    parser.add_argument('--output', type=str, help='Path to save analysis report (JSON)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    print(f"Loading data from: {args.input_file}")
    data = load_d_rollout_data(args.input_file)
    print(f"Loaded {len(data)} entries")
    print()
    
    print_analysis_report(data, args.output)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
