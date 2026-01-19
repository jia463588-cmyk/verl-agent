# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent team.
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

基于论文：Agent learning via Early Experience

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
from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE


class AlternativeActionSampler:
    """使用模型推理（基于 ALFWORLD_TEMPLATE 提示）来采样替代动作。"""
    
    def __init__(self, temperature=1.0, model_path=None):
        """
        初始化替代动作采样器。
        
        参数:
            temperature: 模型推理的采样温度
            model_path: 离线模型的本地路径（必需）
        """
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        
        if not model_path:
            raise ValueError("必须提供 model_path 参数以加载离线模型")
        
        try:
            # 从本地路径加载离线模型
            logging.info(f"从本地路径加载模型: {model_path}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # 检测可用的GPU数量
            gpu_count = torch.cuda.device_count()
            logging.info(f"检测到 {gpu_count} 个可用GPU")
            
            # 从本地路径加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,  # 只使用本地文件，不从网络下载
                trust_remote_code=True
            )
            
            # 根据GPU数量选择设备分配策略
            if gpu_count > 1:
                logging.info(f"使用多GPU模式，将模型分布在 {gpu_count} 个GPU上")
                # 多GPU情况下，使用device_map="auto"自动分配到多个GPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,  # 只使用本地文件
                    trust_remote_code=True,
                    device_map="auto",  # 自动在多个GPU上分配
                    torch_dtype=torch.float16  # 使用半精度以节省显存
                )
            elif gpu_count == 1:
                logging.info("使用单GPU模式")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
            else:
                logging.info("未检测到GPU，使用CPU模式")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
            
            self.model.eval()  # 设置为评估模式
            logging.info("模型加载成功")
            
        except Exception as e:
            logging.error(f"从本地路径加载模型失败: {e}")
            raise RuntimeError(f"无法加载模型，请检查 model_path: {model_path}") from e
    
    def _parse_action_history_and_observation(self, current_state: str) -> Tuple[str, str]:
        """
        从 current_state 解析出动作历史和当前观察。
        
        current_state 格式示例：
        "You have taken the action 1: 'go to coffeemachine 1', action 2: 'take mug 1 from coffeemachine 1' 
         You are now at step 3 and your current observation is: You pick up the mug 1 from the coffeemachine 1."
        
        参数:
            current_state: 完整的当前状态字符串
            
        返回:
            (action_history, current_observation) 元组
        """
        # 查找 "your current observation is:" 分隔符
        obs_marker = "your current observation is:"
        
        if obs_marker in current_state:
            parts = current_state.split(obs_marker, 1)
            action_history = parts[0].strip()
            current_observation = parts[1].strip()
        else:
            # 如果没有动作历史（第一步），整个状态就是当前观察
            action_history = ""
            current_observation = current_state.strip()
        
        return action_history, current_observation
    
    def _construct_prompt(
        self,
        task_description: str,
        step: int,
        current_state: str,
        admissible_commands: List[str]
    ) -> str:
        """
        使用 ALFWORLD_TEMPLATE 构造提示。
        
        参数:
            task_description: 任务描述（从 D_expert 的 task 字段）
            step: 当前步骤编号（从 D_expert 的 step 字段）
            current_state: 完整的当前状态（从 D_expert 的 state_si.current_state 字段）
            admissible_commands: 可执行命令列表（从环境返回）
            
        返回:
            构造的提示字符串
        """
        # 解析动作历史和当前观察
        action_history, current_observation = self._parse_action_history_and_observation(current_state)
        
        # 计算步骤计数（已采取的步骤数）
        step_count = step - 1
        
        # 历史长度等于已采取的步骤数（全历史）
        history_length = step_count
        
        # 当前步骤
        current_step = step
        
        # 格式化可执行命令
        admissible_actions_str = ", ".join(admissible_commands)
        
        # 使用 ALFWORLD_TEMPLATE 构造提示
        prompt = ALFWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=step_count,
            history_length=history_length,
            action_history=action_history,
            current_step=current_step,
            current_observation=current_observation,
            admissible_actions=admissible_actions_str
        )
        
        return prompt
    
    def _extract_action_robust(self, text: str) -> str:
        """
        鲁棒地从生成文本中提取动作。
        
        尝试多种模式以处理模型输出不稳定的情况：
        1. 标准标签: <action>...</action>
        2. 变形标签: [action]...[/action]
        3. 不完整标签: ]go to ...[/action]
        4. 其他格式
        
        参数:
            text: 模型生成的文本
            
        返回:
            提取的动作字符串，如果失败则返回空字符串
        """
        import re
        
        # 模式1: 标准的 <action>...</action>
        match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 模式2: 方括号标签 [action]...[/action]
        match = re.search(r'\[action\]\s*(.*?)\s*\[/action\]', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 模式3: 不完整的开始标签（例如：n]go to table 1[/action]）
        match = re.search(r'\]\s*(.*?)\s*\[/action\]', text, re.IGNORECASE | re.DOTALL)
        if match:
            action_text = match.group(1).strip()
            # 移除可能的前缀字符（如 "n]"）
            if action_text:
                return action_text
        
        # 模式4: 只有结束标签的情况
        match = re.search(r'(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
        if match:
            action_text = match.group(1).strip()
            # 清理可能的残留标签
            action_text = re.sub(r'.*?>', '', action_text).strip()
            if action_text:
                return action_text
        
        # 模式5: 格式 &gt; (HTML实体)
        match = re.search(r'&gt;\s*(.*?)(?:\s|$)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _match_admissible_command(self, text: str, admissible_commands: List[str]) -> str:
        """
        直接在生成文本中查找可执行命令。
        
        当标签提取失败时，尝试在文本中查找任何可执行命令。
        
        参数:
            text: 模型生成的文本
            admissible_commands: 可执行命令列表
            
        返回:
            匹配的命令，如果没有匹配则返回空字符串
        """
        # 规范化文本（转小写）
        text_lower = text.lower()
        
        # 尝试找到最长的匹配命令
        matched_commands = []
        for cmd in admissible_commands:
            if cmd.lower() in text_lower:
                matched_commands.append(cmd)
        
        # 返回最长的匹配（可能更精确）
        if matched_commands:
            return max(matched_commands, key=len)
        
        return ""
    
    def sample_alternative_actions(
        self, 
        task_description: str,
        step: int,
        current_state: str,
        admissible_commands: List[str],
        expert_action: str,
        k: int = 3
    ) -> List[str]:
        """
        使用模型推理采样 k 个不同于专家动作的替代动作。
        
        参数:
            task_description: 任务描述
            step: 当前步骤编号
            current_state: 当前状态文本
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
        
        # 使用模型生成动作
        import torch
        sampled_actions = []
        
        # 使用 ALFWORLD_TEMPLATE 构建提示
        prompt = self._construct_prompt(
            task_description=task_description,
            step=step,
            current_state=current_state,
            admissible_commands=admissible_commands
        )
        
        # 生成 k 个动作
        max_attempts = k * 3  # 最多尝试次数
        attempts = 0
        
        while len(sampled_actions) < k and attempts < max_attempts:
            attempts += 1
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                
                # 从生成的文本中提取动作
                # 尝试多种方式提取动作标签，提高鲁棒性
                generated_action = self._extract_action_robust(generated_text)
                
                # 如果提取失败，尝试直接匹配可执行命令
                if not generated_action:
                    generated_action = self._match_admissible_command(generated_text, alternative_commands)
                
                # 验证生成的动作是否在替代命令中且未被采样过
                if generated_action and generated_action in alternative_commands and generated_action not in sampled_actions:
                    sampled_actions.append(generated_action)
                    logging.debug(f"成功采样替代动作: {generated_action}")
                else:
                    logging.debug(f"生成的动作无效或重复: {generated_action}")
                    
            except Exception as e:
                logging.warning(f"模型推理失败 (尝试 {attempts}): {e}")
        
        if len(sampled_actions) < k:
            logging.warning(f"只成功采样了 {len(sampled_actions)} 个替代动作（目标 {k} 个）")
            # 如果采样不足，从剩余的替代命令中随机选择
            remaining = [cmd for cmd in alternative_commands if cmd not in sampled_actions]
            needed = k - len(sampled_actions)
            if remaining:
                additional = random.sample(remaining, min(needed, len(remaining)))
                sampled_actions.extend(additional)
                logging.info(f"添加了 {len(additional)} 个随机替代动作以达到目标数量")
        
        return sampled_actions


def build_alfworld_env(config_path: str, env_num: int = 1, seed: int = 42, is_train: bool = True):
    """构建 ALFWorld 环境。"""
    # 根据 is_train 参数选择数据集
    if is_train:
        env_kwargs = {}  # 训练模式下使用默认训练数据集
    else:
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
    global_idx = 0  # 初始化全局数据集计数器，从0开始（将在第一次使用时递增到1）
    
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
            batch_size = len(obs['text'])  # 获取批次大小
            
            # 重放前面步骤的专家动作
            for prev_step_entry in expert_steps[:step_num - 1]:
                prev_action = prev_step_entry.get('expert_action_ai', '')
                if prev_action:
                    actions = ["None"] * batch_size
                    actions[env_idx] = prev_action
                    obs, _, _, infos = env_manager.step(actions)
                    info = infos[env_idx]
            
            # 获取当前状态下的可执行命令
            admissible_commands = info.get('admissible_commands', [])
            
            # 采样替代动作（使用模型推理和 ALFWORLD_TEMPLATE）
            alternative_actions = action_sampler.sample_alternative_actions(
                task_description=task_desc,
                step=step_num,
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
                batch_size_branch = len(obs_branch['text'])  # 获取批次大小
                
                # 重放到当前步骤
                for prev_step_entry in expert_steps[:step_num - 1]:
                    prev_action = prev_step_entry.get('expert_action_ai', '')
                    if prev_action:
                        actions_replay = ["None"] * batch_size_branch
                        actions_replay[env_idx] = prev_action
                        obs_branch, _, _, infos_branch = env_manager.step(actions_replay)
                
                # 执行替代动作
                actions_alt = ["None"] * batch_size_branch
                actions_alt[env_idx] = alt_action
                obs_alt, _, _, _ = env_manager.step(actions_alt)
                
                # 获取结果状态
                next_state = obs_alt['text'][env_idx]
                
                # 全局计数器递增
                global_idx += 1
                
                # 以与 D_expert 兼容的格式创建条目
                # idx 字段表示这是数据集中的第几条记录（全局计数）
                rollout_entry = {
                    'task_id': task_id,
                    'idx': global_idx,  # 全局数据集计数，从1开始递增
                    'id': f'rollout_{global_idx:06d}',  # 基于全局计数的唯一ID
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
    parser = argparse.ArgumentParser(description='从专家轨迹生成 D_rollout 数据集（基于论文：Agent learning via Early Experience）')
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
    parser.add_argument('--model_path', type=str, required=True,
                       help='用于动作采样的离线模型的本地路径（必需）')
    parser.add_argument('--is_train', type=lambda x: x.lower() != 'false', default=True,
                       help='是否使用训练数据集（默认：True，设置为False使用评估数据集）')
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
    logging.info(f"构建 ALFWorld 环境... (is_train={args.is_train})")
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config_tw.yaml')
    if not os.path.exists(config_path):
        # 如果相对路径不存在，尝试使用参数中的路径
        config_path = args.config_path
    env_manager = build_alfworld_env(config_path, env_num=1, seed=args.seed, is_train=args.is_train)
    
    # 初始化动作采样器（使用离线模型）
    logging.info(f"初始化动作采样器: model_path={args.model_path}, temperature={args.temperature}")
    action_sampler = AlternativeActionSampler(
        temperature=args.temperature, 
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
