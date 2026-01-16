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
从专家轨迹（D_expert）生成 D_rollout 数据集。

对于专家轨迹中的每个状态 s_i：
1. 采样 K=3 个不同于专家动作的替代动作
2. 使用温度=1.0 的模型推理进行采样
3. 使用环境的可执行命令过滤动作
4. 执行每个替代动作以获取新状态 s_j
5. 存储元组 (s_i, a_j, s_j)，其中 j ∈ [K]

输出格式：D_rollout = {(s_i, a_j, s_j) | i ∈ [N], j ∈ [K]}
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

# 环境和智能体导入
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import alfworld_projection
from agent_system.environments.env_package.alfworld import build_alfworld_envs

# 从 alfworld 包导入专家策略
# 注意：这些需要安装 alfworld 环境包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from alfworld.agents.expert.handcoded_expert_tw import (
    PickAndPlaceSimpleTWPolicy,
    PickTwoObjAndPlaceTWPolicy,
    LookAtObjInLightTWPolicy,
    PickHeatThenPlaceInRecepTWPolicy,
    PickCoolThenPlaceInRecepTWPolicy,
    PickCleanThenPlaceInRecepTWPolicy,
)


class ExpertAgent:
    """使用 ALFWorld 任务的手工编码策略的专家智能体。"""
    
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
        """从游戏文件路径确定任务类型。"""
        for task_type in self.TASK_TYPES.keys():
            if task_type in gamefile:
                return task_type
        return None
    
    def get_policy(self, task_type: str, task_params: dict):
        """获取或创建任务类型的策略。"""
        if task_type not in self.policies:
            policy_class = self.TASK_TYPES.get(task_type)
            if policy_class is None:
                raise ValueError(f"未知的任务类型: {task_type}")
            self.policies[task_type] = policy_class(task_params, self.max_steps)
        return self.policies[task_type]
    
    def act(self, game_state: dict, task_type: str, task_params: dict, last_action: str = ""):
        """获取当前状态的专家动作。"""
        policy = self.get_policy(task_type, task_params)
        return policy.act(game_state, last_action)


class AlternativeActionSampler:
    """使用模型推理或均匀采样来采样替代动作。"""
    
    def __init__(self, temperature=1.0, use_model=False, model_path=None):
        """
        初始化替代动作采样器。
        
        参数:
            temperature: 模型推理的采样温度
            use_model: 是否使用模型进行采样（如果为 False，使用均匀采样）
            model_path: 离线模型的本地路径（如果提供，将从本地加载模型）
        """
        self.temperature = temperature
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        
        if use_model and model_path:
            try:
                # 尝试加载离线模型
                logging.info(f"从本地路径加载模型: {model_path}")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # 从本地路径加载模型和分词器
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,  # 只使用本地文件，不从网络下载
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,  # 只使用本地文件
                    trust_remote_code=True,
                    device_map="auto"  # 自动设备分配
                )
                self.model.eval()  # 设置为评估模式
                logging.info("模型加载成功")
                
            except Exception as e:
                logging.error(f"从本地路径加载模型失败: {e}")
                logging.warning("回退到从可执行命令的均匀采样")
                self.use_model = False
                self.model = None
                self.tokenizer = None
    
    def sample_alternative_actions(
        self, 
        current_state: str,
        admissible_commands: List[str],
        expert_action: str,
        k: int = 3
    ) -> List[str]:
        """
        采样 k 个不同于专家动作的替代动作。
        
        如果启用了模型，将使用模型推理生成动作。
        否则使用从可执行命令的均匀采样。
        
        参数:
            current_state: 当前观察文本
            admissible_commands: 当前状态下的有效动作列表
            expert_action: 专家的动作（需排除）
            k: 要采样的替代动作数量
            
        返回:
            k个替代动作的列表
        """
        # 从可执行命令中过滤掉专家动作
        alternative_commands = [cmd for cmd in admissible_commands if cmd != expert_action]
        
        if len(alternative_commands) == 0:
            logging.warning("没有可用的替代动作，专家动作是唯一选项")
            return []
        
        # 如果使用模型，尝试生成动作
        if self.use_model and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                sampled_actions = []
                
                # 构建提示
                prompt = f"状态: {current_state}\n可用动作: {', '.join(admissible_commands)}\n请选择一个动作："
                
                # 生成 k 个动作
                for _ in range(k):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            temperature=self.temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    generated_action = generated_text.strip()
                    
                    # 验证生成的动作是否在可执行命令中
                    if generated_action in alternative_commands:
                        sampled_actions.append(generated_action)
                    else:
                        # 如果生成的动作无效，从替代命令中随机选择
                        if alternative_commands:
                            sampled_actions.append(random.choice(alternative_commands))
                
                if len(sampled_actions) == k:
                    return sampled_actions
                    
            except Exception as e:
                logging.warning(f"模型推理失败: {e}，回退到均匀采样")
        
        # 均匀采样（默认或回退方法）
        if len(alternative_commands) < k:
            # 替代选项不足，进行有放回采样
            sampled_actions = random.choices(alternative_commands, k=k)
        else:
            # 有足够的替代选项，进行无放回采样
            sampled_actions = random.sample(alternative_commands, k=k)
        
        return sampled_actions
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


def load_expert_trajectories_from_file(filepath: str) -> List[Dict]:
    """
    Load expert trajectories from a JSON file.
    
    Expected format: List of expert trajectory entries, each containing:
    - task_id: unique task identifier
    - idx: trajectory index
    - id: entry id
    - task: task description
    - step: step number
    - state_si: state information with current_state
    - expert_action_ai: expert's action
    - (potentially) next_state_sji: next state after expert action
    - is_expert: True for expert trajectories
    
    Returns:
        List of expert trajectory dictionaries grouped by task_id
    """
    logging.info(f"Loading expert trajectories from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        expert_data = json.load(f)
    
    # Group by task_id to reconstruct trajectories
    trajectories_by_task = defaultdict(list)
    for entry in expert_data:
        task_id = entry.get('task_id', 'unknown')
        trajectories_by_task[task_id].append(entry)
    
    # Sort each trajectory by step number
    trajectories = []
    for task_id, entries in trajectories_by_task.items():
        sorted_entries = sorted(entries, key=lambda x: x.get('step', 0))
        trajectories.append({
            'task_id': task_id,
            'task': sorted_entries[0].get('task', 'unknown'),
            'idx': sorted_entries[0].get('idx', 0),
            'steps': sorted_entries
        })
    
    logging.info(f"Loaded {len(trajectories)} expert trajectories")
    return trajectories


def generate_d_rollout_from_expert_file(
    env_manager,
    expert_trajectories: List[Dict],
    action_sampler: AlternativeActionSampler,
    env_idx: int = 0,
    k: int = 3
) -> List[Dict]:
    """
    Generate D_rollout from pre-collected expert trajectories.
    
    For each step in each expert trajectory:
    1. Set up the environment to that state by replaying expert actions
    2. Sample K alternative actions (different from expert action)
    3. Execute each alternative action to get next_state_sji
    4. Store in D_expert compatible format
    
    Args:
        env_manager: Environment manager
        expert_trajectories: List of expert trajectory dictionaries
        action_sampler: Action sampler for generating alternatives
        env_idx: Environment index to use
        k: Number of alternative actions per state
        
    Returns:
        List of D_rollout entries
    """
    all_d_rollout_entries = []
    
    for traj_data in expert_trajectories:
        task_id = traj_data['task_id']
        task_desc = traj_data['task']
        traj_idx = traj_data['idx']
        expert_steps = traj_data['steps']
        
        logging.info(f"Processing trajectory {traj_idx} (task_id: {task_id}) with {len(expert_steps)} steps")
        
        for step_entry in expert_steps:
            step_num = step_entry.get('step', 1)
            expert_action = step_entry.get('expert_action_ai', '')
            state_si = step_entry.get('state_si', {})
            current_state = state_si.get('current_state', '')
            
            # Reset environment and replay expert actions up to this step
            obs, infos = env_manager.reset({})
            info = infos[env_idx]
            
            # Replay expert actions from previous steps
            for prev_step_entry in expert_steps[:step_num - 1]:
                prev_action = prev_step_entry.get('expert_action_ai', '')
                if prev_action:
                    actions = ["None"] * env_manager.num_processes
                    actions[env_idx] = prev_action
                    obs, _, _, infos = env_manager.step(actions)
                    info = infos[env_idx]
            
            # Get admissible commands at current state
            admissible_commands = info.get('admissible_commands', [])
            
            # Sample alternative actions
            alternative_actions = action_sampler.sample_alternative_actions(
                current_state=current_state,
                admissible_commands=admissible_commands,
                expert_action=expert_action,
                k=k
            )
            
            if len(alternative_actions) == 0:
                logging.debug(f"No alternative actions at step {step_num}, skipping")
                continue
            
            # Execute each alternative action
            for alt_idx, alt_action in enumerate(alternative_actions):
                # Reset and replay again for each alternative
                obs_branch, infos_branch = env_manager.reset({})
                
                # Replay up to current step
                for prev_step_entry in expert_steps[:step_num - 1]:
                    prev_action = prev_step_entry.get('expert_action_ai', '')
                    if prev_action:
                        actions_replay = ["None"] * env_manager.num_processes
                        actions_replay[env_idx] = prev_action
                        obs_branch, _, _, infos_branch = env_manager.step(actions_replay)
                
                # Execute alternative action
                actions_alt = ["None"] * env_manager.num_processes
                actions_alt[env_idx] = alt_action
                obs_alt, _, _, _ = env_manager.step(actions_alt)
                
                # Get resulting state
                next_state = obs_alt['text'][env_idx]
                
                # Create entry in D_expert compatible format
                rollout_entry = {
                    'task_id': task_id,
                    'idx': traj_idx,
                    'id': step_entry.get('id', f'traj_{traj_idx:04d}_step{step_num:03d}') + f'_alt{alt_idx + 1}',
                    'task': task_desc,
                    'step': step_num,
                    'state_si': state_si,
                    'expert_action_ai': expert_action,
                    'alternative_action_j': alt_action,
                    'next_state_sji': next_state,
                    'is_expert': False
                }
                all_d_rollout_entries.append(rollout_entry)
        
        logging.info(f"Generated {len([e for e in all_d_rollout_entries if e['task_id'] == task_id])} rollout entries for trajectory {traj_idx}")
    
    return all_d_rollout_entries


def main():
    parser = argparse.ArgumentParser(description='Generate D_rollout dataset from expert trajectories')
    parser.add_argument('--expert_file', type=str, default=None,
                       help='Path to expert trajectory JSON file (if provided, will use this instead of collecting)')
    parser.add_argument('--config_path', type=str, 
                       default='agent_system/environments/env_package/alfworld/configs/config_tw.yaml',
                       help='Path to ALFWorld config file')
    parser.add_argument('--output_dir', type=str, default='data/d_rollout',
                       help='Output directory for D_rollout dataset')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of expert episodes to collect (only used if --expert_file not provided)')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of alternative actions per state')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode (only used if collecting trajectories)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature for alternative actions')
    parser.add_argument('--use_replay', action='store_true',
                       help='Use replay method to get actual resulting states')
    parser.add_argument('--use_model', action='store_true',
                       help='Use model for action sampling (default: uniform sampling)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Local path to offline model for action sampling')
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
    logging.info("构建 ALFWorld 环境...")
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config_tw.yaml')
    if not os.path.exists(config_path):
        # 如果相对路径不存在，尝试使用参数中的路径
        config_path = args.config_path
    env_manager = build_alfworld_env(config_path, env_num=1, seed=args.seed, is_train=False)
    
    # Initialize action sampler with model support
    logging.info(f"初始化动作采样器: use_model={args.use_model}, model_path={args.model_path}")
    action_sampler = AlternativeActionSampler(
        temperature=args.temperature, 
        use_model=args.use_model,
        model_path=args.model_path
    )
    
    # Generate D_rollout
    all_d_rollout_entries = []
    
    # Check if expert file is provided
    if args.expert_file:
        # Load pre-collected expert trajectories from file
        logging.info(f"Loading expert trajectories from {args.expert_file}")
        
        if not os.path.exists(args.expert_file):
            logging.error(f"Expert file not found: {args.expert_file}")
            return 1
        
        expert_trajectories = load_expert_trajectories_from_file(args.expert_file)
        
        # Generate D_rollout from expert file
        all_d_rollout_entries = generate_d_rollout_from_expert_file(
            env_manager,
            expert_trajectories,
            action_sampler,
            env_idx=0,
            k=args.k
        )
        
        logging.info(f"Generated {len(all_d_rollout_entries)} total rollout entries from expert file")
        
    else:
        # Collect expert trajectories and generate D_rollout (original behavior)
        logging.info("No expert file provided, collecting expert trajectories from scratch...")
        
        # Initialize expert agent
        expert_agent = ExpertAgent(max_steps=args.max_steps)
        
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
        
        # Statistics for collected trajectories
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
    
    # Save D_rollout dataset
    output_file = os.path.join(args.output_dir, 'd_rollout.jsonl')
    logging.info(f"\nSaving D_rollout dataset to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_d_rollout_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save statistics
    if args.expert_file:
        stats = {
            'expert_file': args.expert_file,
            'num_trajectories': len(expert_trajectories) if args.expert_file else 0,
            'total_rollout_entries': len(all_d_rollout_entries),
            'k': args.k,
            'temperature': args.temperature,
        }
    else:
        # stats already defined above in the else block
        pass
    
    stats_file = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n{'='*60}")
    logging.info("D_rollout generation complete!")
    if args.expert_file:
        logging.info(f"Expert file: {args.expert_file}")
        logging.info(f"Trajectories processed: {len(expert_trajectories)}")
    else:
        logging.info(f"Total episodes: {args.num_episodes}")
        logging.info(f"Successful: {successful_episodes}")
        logging.info(f"Failed: {failed_episodes}")
    logging.info(f"Total rollout entries: {len(all_d_rollout_entries)}")
    logging.info(f"Output: {output_file}")
    logging.info(f"Statistics: {stats_file}")


if __name__ == '__main__':
    main()
