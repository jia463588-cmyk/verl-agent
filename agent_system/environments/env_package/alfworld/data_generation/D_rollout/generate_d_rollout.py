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

# 环境导入
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../'))

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import alfworld_projection
from agent_system.environments.env_package.alfworld import build_alfworld_envs


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


def build_alfworld_env(config_path: str, env_num: int = 1, seed: int = 42, is_train: bool = False):
    """构建 ALFWorld 环境。"""
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


def load_expert_trajectories_from_file(filepath: str) -> List[Dict]:
    """
    从 JSON 文件加载专家轨迹。
    
    期望格式：专家轨迹条目列表，每个条目包含：
    - task_id: 唯一任务标识符
    - idx: 轨迹索引
    - id: 条目 ID
    - task: 任务描述
    - step: 步骤编号
    - state_si: 包含 current_state 的状态信息
    - expert_action_ai: 专家的动作
    - (可能) next_state_sji: 专家动作后的下一个状态
    - is_expert: 对于专家轨迹为 True
    
    返回:
        按 task_id 分组的专家轨迹字典列表
    """
    logging.info(f"从 {filepath} 加载专家轨迹")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        expert_data = json.load(f)
    
    # 按 task_id 分组以重建轨迹
    trajectories_by_task = defaultdict(list)
    for entry in expert_data:
        task_id = entry.get('task_id', 'unknown')
        trajectories_by_task[task_id].append(entry)
    
    # 按步骤编号排序每个轨迹
    trajectories = []
    for task_id, entries in trajectories_by_task.items():
        sorted_entries = sorted(entries, key=lambda x: x.get('step', 0))
        trajectories.append({
            'task_id': task_id,
            'task': sorted_entries[0].get('task', 'unknown'),
            'idx': sorted_entries[0].get('idx', 0),
            'steps': sorted_entries
        })
    
    logging.info(f"已加载 {len(trajectories)} 条专家轨迹")
    return trajectories


def generate_d_rollout_from_expert_file(
    env_manager,
    expert_trajectories: List[Dict],
    action_sampler: AlternativeActionSampler,
    env_idx: int = 0,
    k: int = 3
) -> List[Dict]:
    """
    从预收集的专家轨迹生成 D_rollout。
    
    对于每个专家轨迹中的每个步骤：
    1. 通过重放专家动作将环境设置到该状态
    2. 采样 K 个不同于专家动作的替代动作
    3. 执行每个替代动作以获取 next_state_sji
    4. 以与 D_expert 兼容的格式存储
    
    参数:
        env_manager: 环境管理器
        expert_trajectories: 专家轨迹字典列表
        action_sampler: 用于生成替代动作的动作采样器
        env_idx: 要使用的环境索引
        k: 每个状态的替代动作数量
        
    返回:
        D_rollout 条目列表
    """
    all_d_rollout_entries = []
    
    for traj_data in expert_trajectories:
        task_id = traj_data['task_id']
        task_desc = traj_data['task']
        traj_idx = traj_data['idx']
        expert_steps = traj_data['steps']
        
        logging.info(f"处理轨迹 {traj_idx} (task_id: {task_id})，共 {len(expert_steps)} 步")
        
        for step_entry in expert_steps:
            step_num = step_entry.get('step', 1)
            expert_action = step_entry.get('expert_action_ai', '')
            state_si = step_entry.get('state_si', {})
            current_state = state_si.get('current_state', '')
            
            # 重置环境并重放专家动作直到此步骤
            obs, infos = env_manager.reset({})
            info = infos[env_idx]
            
            # 重放前面步骤的专家动作
            for prev_step_entry in expert_steps[:step_num - 1]:
                prev_action = prev_step_entry.get('expert_action_ai', '')
                if prev_action:
                    actions = ["None"] * env_manager.num_processes
                    actions[env_idx] = prev_action
                    obs, _, _, infos = env_manager.step(actions)
                    info = infos[env_idx]
            
            # 获取当前状态下的可执行命令
            admissible_commands = info.get('admissible_commands', [])
            
            # 采样替代动作
            alternative_actions = action_sampler.sample_alternative_actions(
                current_state=current_state,
                admissible_commands=admissible_commands,
                expert_action=expert_action,
                k=k
            )
            
            if len(alternative_actions) == 0:
                logging.debug(f"步骤 {step_num} 没有替代动作，跳过")
                continue
            
            # 执行每个替代动作
            for alt_idx, alt_action in enumerate(alternative_actions):
                # 为每个替代动作重置并重放
                obs_branch, infos_branch = env_manager.reset({})
                
                # 重放到当前步骤
                for prev_step_entry in expert_steps[:step_num - 1]:
                    prev_action = prev_step_entry.get('expert_action_ai', '')
                    if prev_action:
                        actions_replay = ["None"] * env_manager.num_processes
                        actions_replay[env_idx] = prev_action
                        obs_branch, _, _, infos_branch = env_manager.step(actions_replay)
                
                # 执行替代动作
                actions_alt = ["None"] * env_manager.num_processes
                actions_alt[env_idx] = alt_action
                obs_alt, _, _, _ = env_manager.step(actions_alt)
                
                # 获取结果状态
                next_state = obs_alt['text'][env_idx]
                
                # 以与 D_expert 兼容的格式创建条目
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
        
        logging.info(f"为轨迹 {traj_idx} 生成了 {len([e for e in all_d_rollout_entries if e['task_id'] == task_id])} 条 rollout 条目")
    
    return all_d_rollout_entries


def main():
    parser = argparse.ArgumentParser(description='从专家轨迹生成 D_rollout 数据集')
    parser.add_argument('--expert_file', type=str, required=True,
                       help='专家轨迹 JSON 文件的路径')
    parser.add_argument('--config_path', type=str, 
                       default='agent_system/environments/env_package/alfworld/configs/config_tw.yaml',
                       help='ALFWorld 配置文件路径')
    parser.add_argument('--output_dir', type=str, default='data/d_rollout',
                       help='D_rollout 数据集的输出目录')
    parser.add_argument('--k', type=int, default=3,
                       help='每个状态的替代动作数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='替代动作的采样温度')
    parser.add_argument('--use_model', action='store_true',
                       help='使用模型进行动作采样（默认：均匀采样）')
    parser.add_argument('--model_path', type=str, default=None,
                       help='用于动作采样的离线模型的本地路径')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
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
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查专家文件是否存在
    if not os.path.exists(args.expert_file):
        logging.error(f"未找到专家文件: {args.expert_file}")
        return 1
    
    # 构建环境
    logging.info("构建 ALFWorld 环境...")
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config_tw.yaml')
    if not os.path.exists(config_path):
        # 如果相对路径不存在，尝试使用参数中的路径
        config_path = args.config_path
    env_manager = build_alfworld_env(config_path, env_num=1, seed=args.seed, is_train=False)
    
    # 使用模型支持初始化动作采样器
    logging.info(f"初始化动作采样器: use_model={args.use_model}, model_path={args.model_path}")
    action_sampler = AlternativeActionSampler(
        temperature=args.temperature, 
        use_model=args.use_model,
        model_path=args.model_path
    )
    
    # 从文件加载预收集的专家轨迹
    logging.info(f"从 {args.expert_file} 加载专家轨迹")
    expert_trajectories = load_expert_trajectories_from_file(args.expert_file)
    
    # 从专家文件生成 D_rollout
    all_d_rollout_entries = generate_d_rollout_from_expert_file(
        env_manager,
        expert_trajectories,
        action_sampler,
        env_idx=0,
        k=args.k
    )
    
    logging.info(f"从专家文件生成了 {len(all_d_rollout_entries)} 条 rollout 条目")
    
    # 保存 D_rollout 数据集
    output_file = os.path.join(args.output_dir, 'd_rollout.jsonl')
    logging.info(f"\n将 D_rollout 数据集保存到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_d_rollout_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 保存统计信息
    stats = {
        'expert_file': args.expert_file,
        'num_trajectories': len(expert_trajectories),
        'total_rollout_entries': len(all_d_rollout_entries),
        'k': args.k,
        'temperature': args.temperature,
    }
    
    stats_file = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n{'='*60}")
    logging.info("D_rollout 生成完成！")
    logging.info(f"专家文件: {args.expert_file}")
    logging.info(f"处理的轨迹数: {len(expert_trajectories)}")
    logging.info(f"总 rollout 条目数: {len(all_d_rollout_entries)}")
    logging.info(f"输出: {output_file}")
    logging.info(f"统计信息: {stats_file}")


if __name__ == '__main__':
    main()
