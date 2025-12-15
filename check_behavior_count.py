"""
检查分析结果中包含的行为数据数量
"""

import json
import sys

user_uuid = '001841609d1448f18778e70f8c5833df'
result_file = 'intent_result_00184160.json'

try:
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if user_uuid in results:
        user_result = results[user_uuid]
        print(f"用户: {user_uuid}")
        print(f"总会话数: {user_result.get('total_sessions', 0)}")
        print("\n各会话的行为数量:")
        
        total_behaviors = 0
        for session in user_result.get('sessions', []):
            session_idx = session.get('session_index', 0)
            session_size = session.get('session_size', 0)
            total_behaviors += session_size
            print(f"  会话 {session_idx + 1}: {session_size} 个行为")
        
        print(f"\n总计: {total_behaviors} 个行为数据")
        
        # 检查是否有key_behaviors字段
        print("\n关键行为统计:")
        for i, session in enumerate(user_result.get('sessions', []), 1):
            key_behaviors = session.get('key_behaviors', [])
            print(f"  会话 {i}: {len(key_behaviors)} 个关键行为")
            if key_behaviors:
                print(f"    {key_behaviors[:3]}")
    else:
        print(f"未找到用户 {user_uuid} 的分析结果")
        
except FileNotFoundError:
    print(f"未找到结果文件: {result_file}")
    print("请先运行分析: python3 intent_analyzer.py")
except Exception as e:
    print(f"读取文件时出错: {e}")

