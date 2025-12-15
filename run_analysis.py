"""
直接运行用户行为转意图分析
"""

import os
import json
import sys
from intent_analyzer import IntentAnalyzer

def main():
    """主函数"""
    # 检查API密钥
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("错误: 未找到 GEMINI_API_KEY 环境变量")
        print("请先设置: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # 创建分析器
    print("正在初始化分析器...")
    analyzer = IntentAnalyzer(api_key)
    
    # 检查数据文件
    csv_path = 'data.csv'
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        sys.exit(1)
    
    # 询问用户要分析哪些用户
    print("\n" + "="*60)
    print("用户行为转意图分析系统")
    print("="*60)
    print("\n请选择分析模式:")
    print("1. 分析单个用户（输入用户UUID）")
    print("2. 分析前N个用户（输入数字，如10）")
    print("3. 分析所有用户（输入 'all'）")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    # 预加载数据以避免重复读盘
    print(f"\n正在加载数据文件: {csv_path}...")
    import pandas as pd
    df = analyzer.load_data(csv_path)
    
    if choice == '1':
        user_uuid = input("请输入用户UUID: ").strip()
        if not user_uuid:
            print("错误: 用户UUID不能为空")
            sys.exit(1)
        
        user_df = df[df['user_uuid'] == user_uuid]
        if len(user_df) == 0:
            print("错误: 未找到该用户的数据")
            sys.exit(1)
        
        print(f"\n正在分析用户: {user_uuid}")
        print("这可能需要一些时间，请耐心等待...\n")
        
        results = analyzer.analyze_user_intent(
            user_uuid=user_uuid,
            session_timeout_minutes=30,
            preloaded_df=user_df
        )
        
        # 保存结果
        output_file = f'intent_result_{user_uuid[:8]}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析完成！结果已保存到: {output_file}")
        print_results_summary(results)
        
    elif choice == '2':
        try:
            num_users = int(input("请输入要分析的用户数量: ").strip())
        except ValueError:
            print("错误: 请输入有效的数字")
            sys.exit(1)
        
        print(f"\n正在分析前 {num_users} 个用户...")
        print("这可能需要较长时间，请耐心等待...\n")
        
        user_list = df['user_uuid'].unique()[:num_users]
        
        all_results = {}
        for i, user_uuid in enumerate(user_list, 1):
            print(f"[{i}/{len(user_list)}] 正在分析用户: {user_uuid[:8]}...")
            try:
                user_df = df[df['user_uuid'] == user_uuid]
                results = analyzer.analyze_user_intent(
                    user_uuid=user_uuid,
                    session_timeout_minutes=30,
                    preloaded_df=user_df
                )
                all_results.update(results)
            except Exception as e:
                print(f"  警告: 分析用户 {user_uuid[:8]} 时出错: {e}")
                continue
        
        # 保存结果
        output_file = f'intent_result_batch_{num_users}users.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n批量分析完成！结果已保存到: {output_file}")
        print(f"共分析 {len(all_results)} 个用户")
        
    elif choice == '3':
        confirm = input("警告: 分析所有用户可能需要很长时间，确认继续？(yes/no): ").strip().lower()
        if confirm != 'yes':
            print("已取消")
            sys.exit(0)
        
        print("\n正在分析所有用户...")
        print("这可能需要很长时间，请耐心等待...\n")
        
        results = analyzer.analyze_user_intent(
            session_timeout_minutes=30,
            preloaded_df=df
        )
        
        # 保存结果
        output_file = 'intent_result_all.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析完成！结果已保存到: {output_file}")
        print(f"共分析 {len(results)} 个用户")
        
    else:
        print("错误: 无效的选择")
        sys.exit(1)


def print_results_summary(results):
    """打印结果摘要"""
    if not results or 'error' in results:
        print("分析失败或没有结果")
        return
    
    for user_uuid, user_result in results.items():
        if 'sessions' in user_result:
            print(f"\n用户: {user_uuid[:16]}...")
            print(f"  会话数: {user_result.get('total_sessions', 0)}")
            
            for session in user_result.get('sessions', []):
                intent = session.get('intent', 'N/A')
                score = session.get('confidence_score', 0)
                category = session.get('intent_category', 'N/A')
                
                print(f"  会话 {session.get('session_index', 0) + 1}:")
                print(f"    意图: {intent}")
                print(f"    类别: {category}")
                print(f"    置信度: {score:.2f}")


if __name__ == '__main__':
    main()

