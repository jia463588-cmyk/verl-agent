# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate D_rollout dataset from expert trajectories (D_expert).

For each state s_i in expert trajectories:
1. Sample K=3 alternative actions different from the expert action
2. Use model inference with temperature=1.0 for sampling
3. Filter actions using environment admissible commands
4. Execute each alternative action to get new state s_j
5. Store tuples (s_i, a_j, s_j) where j ∈ [K]

Output format: D_rollout = {(s_i, a_j, s_j) | i ∈ [N], j ∈ [K]}
"""

import os
import json
import argparse
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict
import copy
import random

# Environment and agent imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import alfworld_projection
from agent_system.environments.env_package.alfworld import build_alfworld_envs

# Import expert policies from alfworld package
# Note: These need the alfworld environment package to be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
                                '../../agent_system/environments/env_package/alfworld'))
from alfworld.agents.expert.handcoded_expert_tw import (
    PickAndPlaceSimpleTWPolicy,
    PickTwoObjAndPlaceTWPolicy,
    LookAtObjInLightTWPolicy,
    PickHeatThenPlaceInRecepTWPolicy,
    PickCoolThenPlaceInRecepTWPolicy,
    PickCleanThenPlaceInRecepTWPolicy,
)


class ExpertAgent:
    """Expert agent using handcoded policies for ALFWorld tasks."""
    
    TASK_TYPES = {
        'pick_and_place': PickAndPlaceSimpleTWPolicy,
        'pick_two_obj_and_place': PickTwoObjAndPlaceTWPolicy,
        'look_at_obj_in_light': LookAtObjInLightTWPolicy,
        'pick_heat_then_place_in_recep': PickHeatThenPlaceInRecepTWPolicy,
        'pick_cool_then_place_in_recep': PickCoolThenPlaceInRecepTWPolicy,
        'pick_clean_then_place_in_recep': PickCleanThenPlaceInRecepTWPolicy,
    }
    
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.policies = {}
    
    def get_task_type(self, gamefile: str) -> str:
        """Determine task type from gamefile path."""
        for task_type in self.TASK_TYPES.keys():
            if task_type in gamefile:
                return task_type
        return None
    
    def get_policy(self, task_type: str, task_params: dict):
        """Get or create policy for task type."""
        if task_type not in self.policies:
            policy_class = self.TASK_TYPES.get(task_type)
            if policy_class is None:
                raise ValueError(f"Unknown task type: {task_type}")
            self.policies[task_type] = policy_class(task_params, self.max_steps)
        return self.policies[task_type]
    
    def act(self, game_state: dict, task_type: str, task_params: dict, last_action: str = ""):
        """Get expert action for current state."""
        policy = self.get_policy(task_type, task_params)
        return policy.act(game_state, last_action)


class AlternativeActionSampler:
    """Sample alternative actions using model inference or uniform sampling."""
    
    def __init__(self, temperature=1.0, use_model=False, model_path=None):
        """
        Initialize alternative action sampler.
        
        Args:
            temperature: Sampling temperature for model inference
            use_model: Whether to use model for sampling (if False, use uniform sampling)
            model_path: Path to model for inference (optional)
        """
        self.temperature = temperature
        self.use_model = use_model
        self.model = None
        
        if use_model and model_path:
            logging.warning(
                "Model-based sampling not yet implemented. "
                "Falling back to uniform sampling from admissible commands. "
                "To implement: load model/tokenizer and add inference in sample_alternative_actions()"
            )
            self.use_model = False
    
    def sample_alternative_actions(
        self, 
        current_state: str,
        admissible_commands: List[str],
        expert_action: str,
        k: int = 3
    ) -> List[str]:
        """
        Sample k alternative actions different from expert action.
        
        Current implementation uses uniform sampling from admissible commands.
        
        To extend with model-based sampling:
        1. Set use_model=True and provide model_path during initialization
        2. Implement model inference here to generate actions
        3. Validate generated actions against admissible_commands
        4. Fall back to uniform sampling for invalid actions
        
        Args:
            current_state: Current observation text
            admissible_commands: List of valid actions in current state
            expert_action: The expert's action (to exclude)
            k: Number of alternative actions to sample
            
        Returns:
            List of k alternative actions
        """
        # Filter out expert action from admissible commands
        alternative_commands = [cmd for cmd in admissible_commands if cmd != expert_action]
        
        if len(alternative_commands) == 0:
            logging.warning("No alternative actions available, expert action is the only option")
            return []
        
        # Sample k actions (with replacement if necessary)
        if len(alternative_commands) < k:
            # Not enough alternatives, sample with replacement
            sampled_actions = random.choices(alternative_commands, k=k)
        else:
            # Enough alternatives, sample without replacement
            sampled_actions = random.sample(alternative_commands, k=k)
        
        return sampled_actions


def build_alfworld_env(config_path: str, env_num: int = 1, seed: int = 42, is_train: bool = False):
    """Build ALFWorld environment."""
    env_kwargs = {
        'eval_dataset': "eval_in_distribution",
    }
    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}
    group_n = 1
    
    envs = build_alfworld_envs(
        config_path, 
        seed=seed, 
        env_num=env_num, 
        group_n=group_n, 
        is_train=is_train, 
        env_kwargs=env_kwargs, 
        resources_per_worker=resources_per_worker
    )
    env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredTWEnv')
    return env_manager


def collect_expert_trajectory(
    env_manager,
    expert_agent: ExpertAgent,
    env_idx: int = 0,
    max_steps: int = 50
) -> Tuple[List[Dict], bool]:
    """
    Collect a single expert trajectory.
    
    Returns:
        trajectory: List of trajectory steps, each containing:
            - state: observation text
            - expert_action: expert's action
            - admissible_commands: list of valid actions
            - game_state: full game state
            - info: environment info
        success: whether episode succeeded
    """
    trajectory = []
    
    # Reset environment
    obs, infos = env_manager.reset({})
    
    # Get initial info
    info = infos[env_idx]
    gamefile = info.get('extra.gamefile', '')
    task_type = expert_agent.get_task_type(gamefile)
    
    if task_type is None:
        logging.warning(f"Unknown task type for gamefile: {gamefile}")
        return trajectory, False
    
    # Get task parameters from game state
    game_state = {
        'feedback': obs['text'][env_idx],
        'admissible_commands': info.get('admissible_commands', []),
        'facts': info.get('extra.facts', []),
    }
    
    task_params = {
        'object_target': info.get('extra.goal', {}).get('objectType', 'unknown'),
        'parent_target': info.get('extra.goal', {}).get('receptacleType', 'unknown'),
    }
    
    last_action = ""
    done = False
    success = False
    
    for step in range(max_steps):
        # Get expert action
        try:
            expert_action = expert_agent.act(game_state, task_type, task_params, last_action)
        except Exception as e:
            logging.warning(f"Expert failed at step {step}: {e}")
            break
        
        # Store trajectory step
        trajectory_step = {
            'state': obs['text'][env_idx],
            'expert_action': expert_action,
            'admissible_commands': info.get('admissible_commands', []),
            'game_state': copy.deepcopy(game_state),
            'info': copy.deepcopy(info),
            'step': step,
        }
        trajectory.append(trajectory_step)
        
        # Execute expert action
        actions = ["None"] * env_manager.num_processes
        actions[env_idx] = expert_action
        
        obs, rewards, dones, infos = env_manager.step(actions)
        
        # Update state
        done = dones[env_idx]
        info = infos[env_idx]
        success = info.get('won', False)
        
        game_state = {
            'feedback': obs['text'][env_idx],
            'admissible_commands': info.get('admissible_commands', []),
            'facts': info.get('extra.facts', []),
        }
        
        last_action = expert_action
        
        if done:
            break
    
    return trajectory, success


def generate_d_rollout_from_trajectory(
    env_manager,
    trajectory: List[Dict],
    action_sampler: AlternativeActionSampler,
    env_idx: int = 0,
    k: int = 3
) -> List[Dict]:
    """
    Generate D_rollout entries from a single expert trajectory (metadata only).
    
    DEPRECATED: This method only stores metadata without executing actions.
    Use generate_d_rollout_with_replay() instead for accurate state transitions.
    
    For each state s_i in trajectory:
        1. Sample k alternative actions
        2. Store (s_i, a_j, None) where state_j is not computed
    
    This method is kept for reference but use_replay=True is recommended.
    
    Returns:
        List of rollout entries with state_j=None
    """
    logging.warning(
        "Using basic method without replay - state_j will be None. "
        "Consider using generate_d_rollout_with_replay() with --use_replay flag."
    )
    
    d_rollout_entries = []
    
    for traj_step in trajectory:
        state_i = traj_step['state']
        expert_action = traj_step['expert_action']
        admissible_commands = traj_step['admissible_commands']
        step_idx = traj_step['step']
        
        # Sample k alternative actions
        alternative_actions = action_sampler.sample_alternative_actions(
            current_state=state_i,
            admissible_commands=admissible_commands,
            expert_action=expert_action,
            k=k
        )
        
        if len(alternative_actions) == 0:
            logging.debug(f"No alternative actions at step {step_idx}, skipping")
            continue
        
        # Store metadata without executing actions
        for alt_action in alternative_actions:
            rollout_entry = {
                'state_i': state_i,
                'action_j': alt_action,
                'state_j': None,  # Not computed in basic method
                'step': step_idx,
                'expert_action': expert_action,
                'admissible_commands': admissible_commands,
            }
            d_rollout_entries.append(rollout_entry)
    
    return d_rollout_entries


def generate_d_rollout_with_replay(
    env_manager,
    trajectory: List[Dict],
    action_sampler: AlternativeActionSampler,
    env_idx: int = 0,
    k: int = 3,
    task_id: str = None,
    traj_idx: int = 0
) -> List[Dict]:
    """
    Generate D_rollout by replaying trajectory and branching at each step.
    
    This implementation replays the expert trajectory up to each step,
    then executes alternative actions to observe the resulting states.
    
    Note: This method has O(n²k) complexity where n is trajectory length 
    and k is alternatives per step. For large trajectories, consider:
    - Using environment state checkpointing (if available)
    - Processing trajectories in parallel
    - Limiting the number of steps processed
    
    Returns:
        List of rollout entries with actual state transitions in D_expert compatible format
    """
    d_rollout_entries = []
    
    # Extract task information from first trajectory step if available
    if len(trajectory) > 0 and 'info' in trajectory[0]:
        first_info = trajectory[0]['info']
        task_desc = first_info.get('extra.goal_description', 'unknown task')
        gamefile = first_info.get('extra.gamefile', '')
    else:
        task_desc = 'unknown task'
        gamefile = ''
    
    # Generate task_id if not provided
    if task_id is None:
        task_id = f'trial_T{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    
    for step_idx, traj_step in enumerate(trajectory):
        # Reset environment and replay trajectory up to this step
        obs, infos = env_manager.reset({})
        info = infos[env_idx]
        
        # Build action history for state_si
        action_history = []
        for replay_idx in range(step_idx):
            action_history.append(f"action {replay_idx + 1}: '{trajectory[replay_idx]['expert_action']}'")
        
        # Replay expert actions up to current step
        for replay_idx in range(step_idx):
            expert_action = trajectory[replay_idx]['expert_action']
            actions = ["None"] * env_manager.num_processes
            actions[env_idx] = expert_action
            obs, _, _, infos = env_manager.step(actions)
            info = infos[env_idx]
        
        # Now we're at state s_i
        current_obs = obs['text'][env_idx]
        expert_action = traj_step['expert_action']
        admissible_commands = info.get('admissible_commands', [])
        
        # Build current_state with action history
        if action_history:
            current_state = f"You have taken the {', '.join(action_history)}. You are now at step {step_idx + 1} and your current observation is: {current_obs}"
        else:
            current_state = f"You are now at step {step_idx + 1} and your current observation is: {current_obs}"
        
        # Sample alternative actions
        alternative_actions = action_sampler.sample_alternative_actions(
            current_state=current_obs,
            admissible_commands=admissible_commands,
            expert_action=expert_action,
            k=k
        )
        
        # Execute each alternative action
        for alt_idx, alt_action in enumerate(alternative_actions):
            # Need to reset and replay again for each alternative
            obs_branch, infos_branch = env_manager.reset({})
            
            # Replay up to step_idx
            for replay_idx in range(step_idx):
                expert_action_replay = trajectory[replay_idx]['expert_action']
                actions_replay = ["None"] * env_manager.num_processes
                actions_replay[env_idx] = expert_action_replay
                obs_branch, _, _, infos_branch = env_manager.step(actions_replay)
            
            # Execute alternative action
            actions_alt = ["None"] * env_manager.num_processes
            actions_alt[env_idx] = alt_action
            obs_alt, rewards_alt, dones_alt, infos_alt = env_manager.step(actions_alt)
            
            # Get resulting state
            next_state = obs_alt['text'][env_idx]
            
            # Create entry in D_expert compatible format
            rollout_entry = {
                'task_id': task_id,
                'idx': traj_idx,
                'id': f'traj_{traj_idx:04d}_step{step_idx + 1:03d}_alt{alt_idx + 1}',
                'task': task_desc,
                'step': step_idx + 1,
                'state_si': {
                    'current_state': current_state
                },
                'expert_action_ai': expert_action,
                'alternative_action_j': alt_action,
                'next_state_sji': next_state,
                'is_expert': False
            }
            d_rollout_entries.append(rollout_entry)
    
    return d_rollout_entries


def main():
    parser = argparse.ArgumentParser(description='Generate D_rollout dataset from expert trajectories')
    parser.add_argument('--config_path', type=str, 
                       default='agent_system/environments/env_package/alfworld/configs/config_tw.yaml',
                       help='Path to ALFWorld config file')
    parser.add_argument('--output_dir', type=str, default='data/d_rollout',
                       help='Output directory for D_rollout dataset')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of expert episodes to collect')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of alternative actions per state')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature for alternative actions')
    parser.add_argument('--use_replay', action='store_true',
                       help='Use replay method to get actual resulting states')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/generate_d_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build environment
    logging.info("Building ALFWorld environment...")
    config_path = os.path.join(os.path.dirname(__file__), '../../', args.config_path)
    env_manager = build_alfworld_env(config_path, env_num=1, seed=args.seed, is_train=False)
    
    # Initialize expert agent and action sampler
    expert_agent = ExpertAgent(max_steps=args.max_steps)
    action_sampler = AlternativeActionSampler(temperature=args.temperature, use_model=False)
    
    # Collect expert trajectories and generate D_rollout
    all_d_rollout_entries = []
    successful_episodes = 0
    failed_episodes = 0
    
    logging.info(f"Collecting {args.num_episodes} expert trajectories...")
    
    for episode_idx in range(args.num_episodes):
        logging.info(f"\n{'='*60}")
        logging.info(f"Episode {episode_idx + 1}/{args.num_episodes}")
        
        try:
            # Collect expert trajectory
            trajectory, success = collect_expert_trajectory(
                env_manager, 
                expert_agent, 
                env_idx=0,
                max_steps=args.max_steps
            )
            
            if success:
                successful_episodes += 1
                logging.info(f"Expert succeeded in {len(trajectory)} steps")
            else:
                failed_episodes += 1
                logging.info(f"Expert failed/timed out after {len(trajectory)} steps")
            
            # Generate D_rollout entries from trajectory
            if len(trajectory) > 0:
                # Generate unique task_id for this episode
                task_id = f'trial_T{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
                
                if args.use_replay:
                    d_rollout_entries = generate_d_rollout_with_replay(
                        env_manager,
                        trajectory,
                        action_sampler,
                        env_idx=0,
                        k=args.k,
                        task_id=task_id,
                        traj_idx=episode_idx
                    )
                else:
                    d_rollout_entries = generate_d_rollout_from_trajectory(
                        env_manager,
                        trajectory,
                        action_sampler,
                        env_idx=0,
                        k=args.k
                    )
                
                all_d_rollout_entries.extend(d_rollout_entries)
                logging.info(f"Generated {len(d_rollout_entries)} rollout entries")
            
        except Exception as e:
            logging.error(f"Error in episode {episode_idx}: {e}", exc_info=True)
            failed_episodes += 1
    
    # Save D_rollout dataset
    output_file = os.path.join(args.output_dir, 'd_rollout.jsonl')
    logging.info(f"\nSaving D_rollout dataset to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_d_rollout_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats = {
        'num_episodes': args.num_episodes,
        'successful_episodes': successful_episodes,
        'failed_episodes': failed_episodes,
        'total_rollout_entries': len(all_d_rollout_entries),
        'k': args.k,
        'max_steps': args.max_steps,
        'temperature': args.temperature,
        'use_replay': args.use_replay,
    }
    
    stats_file = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n{'='*60}")
    logging.info("D_rollout generation complete!")
    logging.info(f"Total episodes: {args.num_episodes}")
    logging.info(f"Successful: {successful_episodes}")
    logging.info(f"Failed: {failed_episodes}")
    logging.info(f"Total rollout entries: {len(all_d_rollout_entries)}")
    logging.info(f"Output: {output_file}")
    logging.info(f"Statistics: {stats_file}")


if __name__ == '__main__':
    main()
