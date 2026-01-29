#!/usr/bin/env python3
"""
ALFWorld TextWorld 推理测试脚本
仅用于查看 Base Model 在 ALFWorld 环境中的输出

使用的框架文件: 
├── agent_system/environments/env_manager.py          -> AlfWorldEnvironmentManager
├── agent_system/environments/prompts/alfworld.py    -> ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
├── agent_system/environments/env_package/alfworld/
│   ├── __init__. py                                   -> 导出 alfworld_projection, build_alfworld_envs
│   ├── envs.py                                       -> AlfworldEnvs (Ray 并行环境)
│   ├── projection.py                                 -> alfworld_projection (解析 <action> 标签)
│   └── configs/config_tw.yaml                        -> TextWorld 环境配置
└── agent_system/memory/memory.py                     -> SimpleMemory (历史记录管理)
"""

import os
import re
import sys

from sympy import true

# ======================= 路径配置 =======================
PROJECT_ROOT = "/user_home/liangyiwei/Expirement/verl-agent-master"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")  # 仅使用 GPU 5
sys.path.insert(0, PROJECT_ROOT)

# ======================= 导入框架模块 =======================
# 框架文件:  agent_system/environments/env_manager. py
from agent_system.environments.env_manager import AlfWorldEnvironmentManager

# 框架文件:  agent_system/environments/env_package/alfworld/__init__.py
#          -> envs.py, projection.py
from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf

# ======================= 参数配置 =======================
MODEL_PATH = "/user_home/liangyiwei/Expirement/verl-agent-master/offline_Models/Llama-3.1-8B-Instruct"
ENV_NUM = 1          # 只跑1个环境
MAX_STEPS = 5        # 每个环境最多15步，足够看输出
TEMPERATURE = 1
HISTORY_LENGTH = 5   # 历史长度，> 0 时使用 ALFWORLD_TEMPLATE
# 只关注并打印指定环境的输出；设为 None 打印全部
TARGET_ENV = 0

def build_env():
    """
    构建 ALFWorld TextWorld 环境
    
    框架文件:
    - agent_system/environments/env_package/alfworld/configs/config_tw.yaml
    - agent_system/environments/env_package/alfworld/envs.py
    - agent_system/environments/env_manager.py
    """
    # 框架配置文件路径
    alf_config_path = os.path.join(
        PROJECT_ROOT,
        'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    )
    
    env_kwargs = {}
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
    
    # 框架文件: agent_system/environments/env_package/alfworld/envs.py
    envs = build_alfworld_envs(
        alf_config_path,
        seed=42,
        env_num=ENV_NUM,
        group_n=1,
        is_train=true,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker
    )
    # 构建 config 对象，设置 history_length > 0 以使用 ALFWORLD_TEMPLATE
    # 框架文件: agent_system/environments/env_manager.py -> build_text_obs() 
    #          会检查 self.config.env. history_length
    config = OmegaConf.create({
        'env': {
            'history_length':  HISTORY_LENGTH,  # > 0 时使用 ALFWORLD_TEMPLATE
            'env_name': 'alfworld/AlfredTWEnv',
        }
    })
    # 框架文件: agent_system/environments/env_manager.py -> AlfWorldEnvironmentManager
    # 内部使用: agent_system/memory/memory.py -> SimpleMemory
    # 内部使用: agent_system/environments/prompts/alfworld.py -> ALFWORLD_TEMPLATE
    env_manager = AlfWorldEnvironmentManager(
        envs,
        alfworld_projection,  # 框架文件: agent_system/environments/env_package/alfworld/projection.py
        config,
    )
    # 防御性覆盖，确保 history_length 配置可用
    env_manager.config = config
    
    return env_manager


def load_model():
    """加载本地 LLM 模型"""
    print(f"\n[Loading Model] {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("[Model Loaded]\n")
    return model, tokenizer


def generate_action(model, tokenizer, observation):
    """使用模型生成动作"""
    messages = [{"role": "user", "content": observation}]
    
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except: 
        input_text = observation
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    target_device = getattr(model, "device", None)
    if target_device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = "cpu"
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_action_for_display(text:  str) -> str:
    """
    从模型输出中提取 <action>... </action>，仅用于显示目的。
    兼容缺失开头的情况。
    """
    # 标准标签
    m = re.search(r"<action[^>]*>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # 兼容模型缺失 '<' 的情况，如 'action>go to ...</action>'
    m = re.search(r"action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def main():
    print("=" * 80)
    print("ALFWorld TextWorld Inference Test")
    print("=" * 80)
    
    # 构建环境（使用框架）
    print("\n[Building Environment]")
    env_manager = build_env()
    print("[Environment Ready]")
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 重置环境
    # 注意:  第一步 (init=True) 仍会使用 ALFWORLD_TEMPLATE_NO_HIS
    # 从第二步开始使用 ALFWORLD_TEMPLATE（因为需要有历史记录）
    # 框架文件:  agent_system/environments/env_manager. py -> build_text_obs()
    obs, infos = env_manager.reset({})
    env_dones = [False] * ENV_NUM
    
    print("=" * 80)
    print("START INTERACTION")
    print("=" * 80)
    
    for step in range(MAX_STEPS):
        print(f"\n{'='*80}")
        print(f"STEP {step + 1}")
        if step == 0:
            print("(初始步骤，使用 ALFWORLD_TEMPLATE_NO_HIS)")
        else: 
            print(f"(使用 ALFWORLD_TEMPLATE，包含最近 {HISTORY_LENGTH} 步历史)")
        print("=" * 80)
        
        actions = []
        for i in range(ENV_NUM):
            if env_dones[i]: 
                actions.append("None")
                continue
            
            # 获取观察（已由框架格式化）
            # 框架文件: agent_system/environments/env_manager.py -> build_text_obs()
            # step > 0 时使用 ALFWORLD_TEMPLATE（包含历史）
            observation = obs["text"][i]
            
            if TARGET_ENV is None or i == TARGET_ENV:
                print(f"\n[Env {i}] === OBSERVATION ===")
                print(observation)
                print(f"\n[Env {i}] === MODEL RESPONSE ===")
            
            # 模型生成与动作抽取
            raw_response = generate_action(model, tokenizer, observation)
            if TARGET_ENV is None or i == TARGET_ENV:
                print(raw_response)
                parsed_action = extract_action_for_display(raw_response)
                if parsed_action != raw_response:
                    print(f"[Parsed action] {parsed_action}")
            
            actions.append(raw_response)
        
        # 执行动作
        # 框架会使用 alfworld_projection 从 <action>... </action> 提取动作
        # 框架文件: agent_system/environments/env_package/alfworld/projection.py
        obs, rewards, dones, infos = env_manager.step(actions)
        
        # 打印结果
        for i in range(ENV_NUM):
            if not env_dones[i]: 
                if TARGET_ENV is None or i == TARGET_ENV:
                    print(f"\n[Env {i}] Reward: {rewards[i]}, Done:  {dones[i]}, Valid: {infos[i]. get('is_action_valid', 'N/A')}")
                if dones[i]: 
                    env_dones[i] = True
                    if TARGET_ENV is None or i == TARGET_ENV:
                        print(f"[Env {i}] Won:  {infos[i].get('won', False)}")
        
        if all(env_dones):
            print("\n[All environments done]")
            break
    
    print("\n" + "=" * 80)
    print("END")
    print("=" * 80)
    
    env_manager.close()


if __name__ == "__main__": 
    main()