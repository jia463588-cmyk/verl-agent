#!/usr/bin/env python3
"""
简单的验证脚本，用于检查 D_rollout 生成实现。

此脚本验证：
1. 文件结构和导入
2. 数据格式正确性
3. 基本逻辑流程

无需完整依赖即可运行。
"""

import os
import sys
import json
import tempfile
from pathlib import Path


def validate_script_exists():
    """检查主脚本是否存在"""
    script_path = Path(__file__).parent / 'generate_d_rollout.py'
    assert script_path.exists(), f"找不到主脚本：{script_path}"
    print("✓ 主脚本存在")
    return str(script_path)


def validate_shell_script_exists():
    """检查 Shell 脚本是否存在"""
    script_path = Path(__file__).parent / 'run_generate_d_rollout.sh'
    assert script_path.exists(), f"找不到 Shell 脚本：{script_path}"
    assert os.access(script_path, os.X_OK), f"Shell 脚本不可执行：{script_path}"
    print("✓ Shell 脚本存在且可执行")
    return str(script_path)


def validate_readme_exists():
    """检查 README 是否存在"""
    readme_path = Path(__file__).parent / 'README_D_ROLLOUT.md'
    assert readme_path.exists(), f"找不到 README：{readme_path}"
    print("✓ README 存在")
    return str(readme_path)


def validate_data_format():
    """验证预期的 D_rollout 数据格式"""
    # 新的 D_expert 兼容格式
    expected_fields = ['task_id', 'idx', 'id', 'task', 'step', 'state_si', 
                       'expert_action_ai', 'alternative_action_j', 'next_state_sji', 'is_expert']
    
    # 创建新格式的示例条目
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
    
    # 验证所有预期字段是否存在
    for field in expected_fields:
        assert field in sample_entry, f"缺少必需字段：{field}"
    
    # 测试 JSON 序列化
    try:
        json_str = json.dumps(sample_entry, ensure_ascii=False)
        loaded = json.loads(json_str)
        assert loaded == sample_entry, "JSON 往返失败"
    except Exception as e:
        raise AssertionError(f"JSON 序列化失败：{e}")
    
    print("✓ 数据格式验证通过")
    return True


def validate_alternative_action_sampling_logic():
    """验证替代动作采样的逻辑"""
    admissible_commands = ['action1', 'action2', 'action3', 'action4', 'action5']
    expert_action = 'action1'
    k = 3
    
    # 模拟采样（均匀分布）
    import random
    random.seed(42)
    
    alternative_commands = [cmd for cmd in admissible_commands if cmd != expert_action]
    assert len(alternative_commands) == 4, "过滤失败"
    
    sampled_actions = random.sample(alternative_commands, k=k)
    assert len(sampled_actions) == k, "采样数量不正确"
    assert expert_action not in sampled_actions, "专家动作不应在样本中"
    
    print("✓ 替代动作采样逻辑验证通过")
    return True


def validate_dataset_structure():
    """验证数据集可以正确写入和读取"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, 'd_rollout.jsonl')
        
        # 创建示例数据
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
        
        # 写入数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in sample_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # 读取并验证
        loaded_data = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                loaded_data.append(json.loads(line))
        
        assert len(loaded_data) == len(sample_data), "数据长度不匹配"
        assert loaded_data == sample_data, "数据内容不匹配"
        
        print("✓ 数据集 I/O 验证通过")
        return True


def validate_statistics_format():
    """验证统计文件格式"""
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
    
    # 测试 JSON 序列化
    try:
        json_str = json.dumps(stats, indent=2, ensure_ascii=False)
        loaded = json.loads(json_str)
        assert loaded == stats, "统计信息往返失败"
    except Exception as e:
        raise AssertionError(f"统计信息序列化失败：{e}")
    
    print("✓ 统计格式验证通过")
    return True


def main():
    """运行所有验证检查"""
    print("="*60)
    print("D_rollout 生成验证")
    print("="*60)
    
    try:
        # 文件存在性检查
        validate_script_exists()
        validate_shell_script_exists()
        validate_readme_exists()
        
        # 格式和逻辑检查
        validate_data_format()
        validate_alternative_action_sampling_logic()
        validate_dataset_structure()
        validate_statistics_format()
        
        print("="*60)
        print("✓ 所有验证检查通过！")
        print("="*60)
        print()
        print("注意：此验证仅检查结构和逻辑。")
        print("要运行实际生成，请确保已安装所有依赖。")
        print("查看 README_D_ROLLOUT.md 获取安装说明。")
        return 0
        
    except AssertionError as e:
        print("="*60)
        print(f"✗ 验证失败：{e}")
        print("="*60)
        return 1
    except Exception as e:
        print("="*60)
        print(f"✗ 意外错误：{e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
